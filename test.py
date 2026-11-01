import torch
import numpy as np
import pandas as pd
import os
import time
import shutil
import threading
import platform
import subprocess
from pcc_model import PCCModel
from coder import Coder
from data_utils import load_sparse_tensor, sort_spare_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from pc_error import pc_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    import psutil
except Exception:
    psutil = None

NAN = float('nan')
IS_WSL = 'microsoft' in platform.release().lower()
IS_WIN = (os.name == 'nt')
_trapz = getattr(np, 'trapezoid', None) or np.trapz


# =====================================================================
#  OUTILS AJOUTÉS : mémoire, MACs, énergie (EnergiBridge)
# =====================================================================

def _run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return r.stdout.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def _to_win_path(p):
    """Chemin utilisable par un .exe Windows (wslpath -w sous WSL)."""
    if IS_WSL:
        w = _run(['wslpath', '-w', p]).strip()
        return w if w else p
    return p


def _clock_skew():
    """Décalage (horloge Windows - horloge Python) en secondes. 0 en Windows natif."""
    if not IS_WSL:
        return 0.0
    t_a = time.time()
    out = _run(['powershell.exe', '-NoProfile', '-Command',
                '[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()'])
    t_b = time.time()
    try:
        return float(out.strip()) / 1000.0 - (t_a + t_b) / 2.0
    except Exception:
        print('[EB] Impossible de synchroniser les horloges, décalage supposé nul.')
        return 0.0


def _read_rss_mb():
    """RSS courant du processus en Mo (Linux/WSL via /proc, sinon psutil)."""
    try:
        with open('/proc/self/statm') as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE') / (1024 ** 2)
    except Exception:
        pass
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except Exception:
            pass
    return NAN


def _read_vmhwm_mb():
    """Pic de RSS du processus (VmHWM) en Mo (Linux/WSL uniquement)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return NAN


def _fin(*vals):
    return all(np.isfinite(v) for v in vals)


class ResourceMonitor:
    """Mesure le temps et le pic RSS du processus pendant une phase."""

    def __enter__(self):
        self.reset_ok = False

        try:
            with open('/proc/self/clear_refs', 'w') as f:
                f.write('5')
            self.reset_ok = True
        except Exception:
            pass

        self.rss0 = _read_rss_mb()
        self.t0 = time.time()
        self.perf0 = time.perf_counter()

        return self

    def __exit__(self, *exc):
        self.t1 = time.time()
        self.elapsed = time.perf_counter() - self.perf0

        hwm = _read_vmhwm_mb() if self.reset_ok else NAN

        if np.isfinite(hwm):
            self.peak_rss = hwm
        else:
            self.peak_rss = _read_rss_mb()

        if np.isfinite(self.peak_rss) and np.isfinite(self.rss0):
            self.mem_delta = max(0.0, self.peak_rss - self.rss0)
        else:
            self.mem_delta = NAN

        return False

class MacsCounter:
    """MACs = somme sur les convolutions Minkowski de n_out * kernel_volume * Cin * Cout
    (borne haute, poids denses). Compté par hooks pendant encode() puis decode()."""
    def __init__(self, model):
        self.total = 0.0
        for name, m in model.named_modules():
            if 'Convolution' in type(m).__name__ and hasattr(m, 'kernel_generator'):
                m.register_forward_hook(self._hook)

    def reset(self):
        self.total = 0.0

    def _hook(self, module, inp, out):
        try:
            n_out = out.F.shape[0]
            kv = module.kernel_generator.kernel_volume
            self.total += float(n_out) * kv * module.in_channels * module.out_channels
        except Exception:
            pass


class EnergiBridgeRecorder:
    """Lance EnergiBridge en arrière-plan pendant tout le test (un seul enregistrement continu),
    puis calcule l'énergie sur les fenêtres [t0, t1] de chaque encodage / décodage uniquement."""

    def __init__(self, tag, exe=None, interval=None, column=None, energy_scale=1.0, max_seconds=7200):
        self.available = False
        self.proc = None
        self.tag = tag
        self.interval = interval
        self.column_forced = column
        self.scale = energy_scale
        self.max_seconds = max_seconds
        self.kind = None          # 'cumulative' (J) ou 'power' (W)
        self.column = None
        self.skew = 0.0
        self.ts = np.array([])
        self.val = np.array([])
        self.csv_path = None
        self.exe = exe or shutil.which('energibridge') or shutil.which('energibridge.exe')
        if self.exe is None:
            print("[EB] energibridge introuvable dans le PATH. Utilise --eb_path <chemin complet de energibridge.exe>.")
            return
        if IS_WSL:
            self.dir = '/mnt/c/Users/Public/dynsparse_eb'      # dossier côté Windows, sans espaces
        elif IS_WIN:
            self.dir = r'C:\Users\Public\dynsparse_eb'
        else:
            self.dir = os.path.abspath('./dynsparse_eb')
        os.makedirs(self.dir, exist_ok=True)
        self.available = True

    def start(self):
        if not self.available:
            return
        self.available = False
        self.csv_path = os.path.join(self.dir, self.tag + '_energibridge.csv')
        self.stop_path = os.path.join(self.dir, self.tag + '.stop')
        self.err_path = os.path.join(self.dir, self.tag + '_eb_stderr.txt')
        bat_path = os.path.join(self.dir, 'eb_wait.bat')
        for p in (self.csv_path, self.stop_path):
            if os.path.exists(p):
                os.remove(p)
        # petit script batch : attend le fichier "stop", ce qui permet à EnergiBridge
        # de se terminer proprement et de vider son CSV sur disque.
        bat = ('@echo off\r\nset /a i=0\r\n:loop\r\nif exist "%~1" goto end\r\n'
               'if %i% GEQ ' + str(self.max_seconds) + ' goto end\r\n'
               'ping -n 2 127.0.0.1 >nul\r\nset /a i+=1\r\ngoto loop\r\n:end\r\n')
        with open(bat_path, 'w', newline='') as f:
            f.write(bat)
        self.skew = _clock_skew()
        cmd = [self.exe, '-o', _to_win_path(self.csv_path)]
        if self.interval:
            cmd += ['-i', str(self.interval)]
        cmd += ['--', 'cmd.exe', '/c', _to_win_path(bat_path), _to_win_path(self.stop_path)]
        self._err = open(self.err_path, 'wb')
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self._err)
        time.sleep(4)
        if self.proc.poll() is not None:
            print("[EB] EnergiBridge s'est arrêté au démarrage (code %s). Lance le terminal EN ADMINISTRATEUR. "
                  "Détails: %s" % (self.proc.returncode, self.err_path))
            return
        self.available = True
        print('[EB] enregistrement en cours ->', self.csv_path, '| décalage horloge: %.3f s' % self.skew)

    def stop(self):
        if self.proc is None:
            return
        time.sleep(2)                       # laisse arriver le dernier échantillon
        try:
            open(self.stop_path, 'w').close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.terminate()
        try:
            self._err.close()
        except Exception:
            pass

    def load(self):
        if not self.available or not os.path.exists(self.csv_path):
            self.available = False
            return
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print('[EB] CSV illisible:', e)
            self.available = False
            return
        df.columns = [str(c).strip() for c in df.columns]
        tcol = next((c for c in df.columns if c.lower() == 'time'), None)
        if tcol is None:
            print('[EB] colonne Time absente. Colonnes:', list(df.columns))
            self.available = False
            return
        cols = list(df.columns)
        if self.column_forced:
            col = self.column_forced
            kind = 'cumulative' if 'ENERGY' in col.upper() else 'power'
        else:
            cum = [c for c in cols if 'ENERGY' in c.upper() and 'PACKAGE' in c.upper()]
            if not cum:
                cum = [c for c in cols if 'ENERGY' in c.upper() and 'DRAM' not in c.upper()]
            pw = [c for c in cols if 'POWER' in c.upper()]
            pw = sorted(pw, key=lambda c: (0 if 'PACKAGE' in c.upper() else 1 if 'CPU' in c.upper() else 2))
            if cum:
                col, kind = cum[0], 'cumulative'
            elif pw:
                col, kind = pw[0], 'power'
            else:
                print('[EB] Aucune colonne ENERGY/POWER trouvée. Colonnes disponibles:')
                print(cols)
                self.available = False
                return
        self.column, self.kind = col, kind
        d = df[[tcol, col]].apply(pd.to_numeric, errors='coerce').dropna()
        ts = d[tcol].values / 1000.0 - self.skew         # epoch ms -> s, ramené sur l'horloge Python
        v = d[col].values.astype(float) * self.scale
        if kind == 'cumulative':
            dv = np.diff(v, prepend=v[0])
            dv[dv < 0] = 0.0                              # protège contre un rebouclage du compteur
            v = np.cumsum(dv)
        self.ts, self.val = ts, v
        print('[EB] colonne utilisée: "%s" (%s) | %d échantillons | pas médian: %.0f ms'
              % (col, 'énergie cumulée J' if kind == 'cumulative' else 'puissance W',
                 len(ts), 1000 * float(np.median(np.diff(ts))) if len(ts) > 1 else NAN))
        print('[EB] autres colonnes ENERGY/POWER vues:',
              [c for c in cols if ('ENERGY' in c.upper() or 'POWER' in c.upper()) and c != col])
        if len(ts) < 2:
            self.available = False

    def energy_j(self, t0, t1):
        """Énergie (J) consommée entre t0 et t1."""
        if not self.available or len(self.ts) < 2 or t1 <= t0 or t0 < self.ts[0] or t1 > self.ts[-1]:
            return NAN
        if self.kind == 'cumulative':
            return float(np.interp(t1, self.ts, self.val) - np.interp(t0, self.ts, self.val))
        inner = (self.ts > t0) & (self.ts < t1)
        tt = np.concatenate([[t0], self.ts[inner], [t1]])
        pp = np.concatenate([[np.interp(t0, self.ts, self.val)], self.val[inner],
                             [np.interp(t1, self.ts, self.val)]])
        return float(_trapz(pp, tt))

    def n_samples(self, t0, t1):
        return int(((self.ts >= t0) & (self.ts <= t1)).sum()) if len(self.ts) else 0


# =====================================================================
#  TEST (code d'origine + mémoire / énergie / MACs)
# =====================================================================

def test(filedir, ckptdir_list, outdir, resultdir, scaling_factor=1.0, rho=1.0, res=1024,
         idle_seconds=0, energy_source='energibridge', eb_path=None, eb_interval=None,
         eb_column=None, energy_scale=1.0):
    # load data
    start_time = time.time()
    x = load_sparse_tensor(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')
    # x = sort_spare_tensor(input_data)

    # output filename
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
    print('output filename:\t', filename)

    # load model
    model = PCCModel().to(device)

    # --- ajout : compteur de MACs et enregistrement d'énergie ---
    scene = os.path.split(filedir)[-1].split('.')[0]
    macs = MacsCounter(model)
    rec = None
    if energy_source == 'energibridge':
        rec = EnergiBridgeRecorder(scene, exe=eb_path, interval=eb_interval,
                                   column=eb_column, energy_scale=energy_scale)
        rec.start()

    # puissance au repos (optionnelle, uniquement pour les colonnes *_net_J)
    idle_t0 = idle_t1 = None
    if idle_seconds > 0 and rec is not None and rec.available:
        print('Mesure de la puissance au repos pendant', idle_seconds, 's (ne touche pas au PC)...')
        idle_t0 = time.time()
        time.sleep(idle_seconds)
        idle_t1 = time.time()

    runs_store = []

    for idx, ckptdir in enumerate(ckptdir_list):
        print('='*10, idx+1, '='*10)
        # load checkpoints
        assert os.path.exists(ckptdir)
        ckpt = torch.load(ckptdir)
        model.load_state_dict(ckpt['model'])
        print('load checkpoint from \t', ckptdir)
        coder = Coder(model=model, filename=filename)

        # postfix: rate index
        postfix_idx = '_r'+str(idx+1)

        # down-scale
        if scaling_factor!=1: 
            x_in = scale_sparse_tensor(x, factor=scaling_factor)
        else: 
            x_in = x

        # encode
        macs.reset()
        start_time = time.time()
        with ResourceMonitor() as mon_enc:
            _ = coder.encode(x_in, postfix=postfix_idx)
        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)
        macs_enc = macs.total

        # decode
        macs.reset()
        start_time = time.time()
        with ResourceMonitor() as mon_dec:
            x_dec = coder.decode(postfix=postfix_idx, rho=rho)
        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)
        macs_dec = macs.total

        # up-scale
        if scaling_factor!=1: 
            x_dec = scale_sparse_tensor(x_dec, factor=1.0/scaling_factor)

        # bitrate
        bits = np.array([os.path.getsize(filename + postfix_idx + postfix)*8 \
                                for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
        bpps = (bits/len(x)).round(3)
        print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

        # distortion
        start_time = time.time()
        write_ply_ascii_geo(filename+postfix_idx+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

        start_time = time.time()
        pc_error_metrics = pc_error(filedir, filename+postfix_idx+'_dec.ply', 
                                    res=res, normal=True, show=False)
        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])

        # save results
        results = pc_error_metrics
        results["num_points(input)"] = len(x)
        results["num_points(output)"] = len(x_dec)
        results["resolution"] = res
        results["bits"] = sum(bits).round(3)
        results["bpp"] = sum(bpps).round(3)
        results["bpp(coords)"] = bpps[0]
        results["bpp(feats)"] = bpps[1]
        results["time(enc)"] = time_enc
        results["time(dec)"] = time_dec

        # ---------- colonnes ajoutées : mémoire ----------
        results["mem_peak_rss(enc)_MB"] = round(mon_enc.peak_rss, 1)
        results["mem_peak_rss(dec)_MB"] = round(mon_dec.peak_rss, 1)
        results["mem_peak_rss(codec)_MB"] = round(max(mon_enc.peak_rss, mon_dec.peak_rss), 1)
        results["mem_delta(enc)_MB"] = round(mon_enc.mem_delta, 1)
        results["mem_delta(dec)_MB"] = round(mon_dec.mem_delta, 1)
        results["mem_delta(codec)_MB"] = round(max(mon_enc.mem_delta, mon_dec.mem_delta), 1)

        # ---------- colonnes ajoutées : MACs ----------
        results["MACs(enc)"] = macs_enc
        results["MACs(dec)"] = macs_dec
        results["MACs(codec)"] = macs_enc + macs_dec
        results["MACs_per_point(enc)"] = round(macs_enc / len(x), 1)
        results["MACs_per_point(dec)"] = round(macs_dec / len(x), 1)

        # ---------- colonnes ajoutées : énergie (remplies après la boucle) ----------
        for c in ["energy(enc)_J", "energy(dec)_J", "energy(codec)_J",
                  "avg_power(enc)_W", "avg_power(dec)_W", "avg_power(codec)_W",
                  "idle_power_W", "energy(enc)_net_J", "energy(dec)_net_J", "energy(codec)_net_J"]:
            results[c] = NAN
        results["energy_method"] = ''
        results["energy_column"] = ''

        runs_store.append((idx, (mon_enc.t0, mon_enc.t1), (mon_dec.t0, mon_dec.t1)))

        if idx == 0:
            all_results = results.copy(deep=True)
        else: 
            all_results = pd.concat([all_results, results], ignore_index=True)
        csv_name = os.path.join(resultdir, os.path.split(filedir)[-1].split('.')[0]+'.csv')
        all_results.to_csv(csv_name, index=False)   # sauvegarde intermédiaire (énergie encore NaN)
        print('Write results to: \t', csv_name)

    # ------------------------------------------------------------------
    # Fin des mesures : arrêt d'EnergiBridge, puis énergie par phase
    # ------------------------------------------------------------------
    energy_ok = False
    idle_w = NAN
    if rec is not None:
        rec.stop()
        rec.load()
        energy_ok = rec.available
    if energy_ok:
        if idle_t0 is not None:
            e_idle = rec.energy_j(idle_t0 + 2.0, idle_t1)
            idle_w = e_idle / (idle_t1 - idle_t0 - 2.0) if np.isfinite(e_idle) else NAN
            print('Puissance au repos: %.3f W' % idle_w)
        try:
            logdir = os.path.join(resultdir, 'logs')
            if not os.path.exists(logdir): os.makedirs(logdir)
            shutil.copy(rec.csv_path, os.path.join(logdir, scene + '_energibridge.csv'))
        except Exception:
            pass
    else:
        print("\nATTENTION : aucune mesure d'énergie valide. Les colonnes d'énergie restent NaN "
              "(pas d'estimation).")

    for idx, (e0, e1), (d0, d1) in runs_store:
        if not energy_ok:
            continue
        E_enc = rec.energy_j(e0, e1)
        E_dec = rec.energy_j(d0, d1)
        for name, n in (('enc', rec.n_samples(e0, e1)), ('dec', rec.n_samples(d0, d1))):
            if n < 3:
                print("ATTENTION: seulement %d échantillon(s) EnergiBridge pour rate %d / %s "
                      "(phase courte : baisse --eb_interval)" % (n, idx + 1, name))
        E_codec = E_enc + E_dec
        t_enc, t_dec = e1 - e0, d1 - d0
        all_results.loc[idx, "energy(enc)_J"] = E_enc
        all_results.loc[idx, "energy(dec)_J"] = E_dec
        all_results.loc[idx, "energy(codec)_J"] = E_codec
        all_results.loc[idx, "avg_power(enc)_W"] = E_enc / t_enc if _fin(E_enc) and t_enc > 0 else NAN
        all_results.loc[idx, "avg_power(dec)_W"] = E_dec / t_dec if _fin(E_dec) and t_dec > 0 else NAN
        all_results.loc[idx, "avg_power(codec)_W"] = E_codec / (t_enc + t_dec) if _fin(E_codec) else NAN
        if _fin(idle_w):
            all_results.loc[idx, "idle_power_W"] = idle_w
            all_results.loc[idx, "energy(enc)_net_J"] = E_enc - idle_w * t_enc
            all_results.loc[idx, "energy(dec)_net_J"] = E_dec - idle_w * t_dec
            all_results.loc[idx, "energy(codec)_net_J"] = E_codec - idle_w * (t_enc + t_dec)
        all_results.loc[idx, "energy_method"] = "EnergiBridge"
        all_results.loc[idx, "energy_column"] = rec.column
        print("Rate", idx + 1, "| Enc: %.3f J" % E_enc, "| Dec: %.3f J" % E_dec,
              "| Codec: %.3f J" % E_codec)

    csv_name = os.path.join(resultdir, os.path.split(filedir)[-1].split('.')[0]+'.csv')
    all_results.to_csv(csv_name, index=False)
    print('Résultats finaux sauvegardés :', csv_name)

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='../../../testdata/8iVFB/longdress_vox10_1300.ply')
    parser.add_argument("--outdir", default='./output')
    parser.add_argument("--resultdir", default='./results')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    # --- options ajoutées ---
    parser.add_argument("--energy_source", default='energibridge', choices=['energibridge', 'none'])
    parser.add_argument("--eb_path", default=None, help='chemin complet de energibridge(.exe) si absent du PATH')
    parser.add_argument("--eb_interval", type=int, default=None, help="option -i d'EnergiBridge (voir energibridge --help)")
    parser.add_argument("--eb_column", default=None, help='forcer la colonne énergie/puissance du CSV EnergiBridge')
    parser.add_argument("--energy_scale", type=float, default=1.0, help='facteur pour convertir la colonne en J (ou W)')
    parser.add_argument("--idle_seconds", type=int, default=0, help='>0 : mesure la puissance au repos (s) pour les colonnes *_net_J')
    parser.add_argument("--max_rates", type=int, default=0, help='0 = les 7 débits ; N = seulement les N premiers (test rapide)')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    ckptdir_list = ['./ckpts/r1_0.025bpp.pth', './ckpts/r2_0.05bpp.pth', 
                    './ckpts/r3_0.10bpp.pth', './ckpts/r4_0.15bpp.pth', 
                    './ckpts/r5_0.25bpp.pth', './ckpts/r6_0.3bpp.pth', 
                    './ckpts/r7_0.4bpp.pth']
    if args.max_rates > 0:
        ckptdir_list = ckptdir_list[:args.max_rates]

    all_results = test(args.filedir, ckptdir_list, args.outdir, args.resultdir,
                       scaling_factor=args.scaling_factor, rho=args.rho, res=args.res,
                       idle_seconds=args.idle_seconds, energy_source=args.energy_source,
                       eb_path=args.eb_path, eb_interval=args.eb_interval,
                       eb_column=args.eb_column, energy_scale=args.energy_scale)

    # plot RD-curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
            label="D1", marker='x', color='red')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
            label="D2", marker='x', color='blue')
    filename = os.path.split(args.filedir)[-1][:-4]
    plt.title(filename)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(args.resultdir, filename+'.jpg'))