"""
test.py -- PCGCv2 (NJUVISION) : codec ORIGINAL + mesures reelles pour le projet
DynSparse-PCC.


Metriques d'origine (inchangees) :
    pc_error (D1/D2...), num_points, resolution, bits, bpp,
    bpp(coords), bpp(feats), time(enc), time(dec).


Metriques ajoutees pour la comparaison (cahier des charges) :
    - Latence : latency(codec)[s]
    - Energie reelle CPU : EnergiBridge / RAPL PACKAGE_ENERGY
        energy(enc)[J], energy(dec)[J], energy(codec)[J]
        power_avg(enc)[W], power_avg(dec)[W]
    - Memoire reelle : pic RSS python + processus fils
        mem_peak(codec)[MB]
    - Cout theorique : MACs des convolutions sparses
        MACs(enc), MACs(dec), MACs(codec)
    - Modele : params, params_nonzero


Pruning ajoute :
    - pruning_mode = none / static / dynamic
    - static : sparsity globale identique pour tous les rates
    - dynamic : deux niveaux de sparsite selon la densite locale des tuiles

L'energie est mesuree separement pour l'encodage et le decodage,
avec une instance EnergiBridge dediee a chaque rate.
"""


import torch
import numpy as np
import pandas as pd
import os
import sys
import time
import signal
import select
import threading
import subprocess
import shutil
from pcc_model import PCCModel
from coder import Coder
from data_utils import load_sparse_tensor, sort_spare_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from pc_error import pc_error

from pruning.static_pruning import apply_static_pruning
from pruning.dynamic_pruning import apply_dynamic_pruning
from pruning.pruning_utils import save_json


try:
    import psutil
except ImportError:
    psutil = None


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NAN = float('nan')
MB = 1024.0 * 1024.0



# =====================================================================================
#  MEMOIRE
# =====================================================================================
class MemProbe(object):
    """Mesure le pic memoire (RSS) pendant un bloc `with`."""


    def __init__(self, period=0.01):
        if psutil is None:
            raise RuntimeError("psutil est requis : pip install psutil")
        self.period = period
        self.proc = psutil.Process(os.getpid())


    def __enter__(self):
        self.peak_total = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self


    def _loop(self):
        while not self._stop.is_set():
            try:
                tot = self.proc.memory_info().rss


                for c in self.proc.children(recursive=True):
                    try:
                        tot += c.memory_info().rss
                    except psutil.Error:
                        pass


                if tot > self.peak_total:
                    self.peak_total = tot


            except psutil.Error:
                pass


            self._stop.wait(self.period)


    def __exit__(self, *a):
        self._stop.set()
        self._th.join()
        return False


    @property
    def peak_total_mb(self):
        return round(self.peak_total / MB, 2)



# =====================================================================================
#  MACs (convolutions sparses MinkowskiEngine)
# =====================================================================================
class MacCounter(object):
    """MACs theoriques des convolutions sparses.


    MACs = somme_couches( paires * C_in * C_out ).
    """


    def __init__(self, model):
        import MinkowskiEngine as ME


        types = (
            ME.MinkowskiConvolution,
            ME.MinkowskiConvolutionTranspose,
            ME.MinkowskiGenerativeConvolutionTranspose
        )


        self.records = []
        self.handles = [
            m.register_forward_hook(self._hook)
            for m in model.modules() if isinstance(m, types)
        ]
        self.warned = False


    def _hook(self, module, inputs, output):
        self.records.append((
            module,
            inputs[0].coordinate_map_key,
            output.coordinate_map_key,
            output.coordinate_manager
        ))


    def reset(self):
        self.records = []


    def compute(self):
        total = 0


        try:
            for module, kin, kout, cm in self.records:
                kg = module.kernel_generator


                kmap = cm.kernel_map(
                    kin, kout,
                    stride=kg.kernel_stride,
                    kernel_size=kg.kernel_size,
                    dilation=kg.kernel_dilation,
                    region_type=kg.region_type,
                    region_offset=kg.region_offsets,
                    is_transpose=module.is_transpose,
                    is_pool=False
                )


                pairs = sum(int(v[0].numel()) for v in kmap.values())
                total += pairs * int(module.in_channels) * int(module.out_channels)


        except Exception as e:
            if not self.warned:
                print('[MACs] comptage impossible (%s) -> NaN' % e)
                self.warned = True
            total = NAN


        self.records = []
        return float(total)



# =====================================================================================
#  ENERGIE : EnergiBridge (Windows) pilote depuis WSL
# =====================================================================================
def _readline_fd(fd, timeout):
    buf = b''
    end = time.time() + timeout


    while b'\n' not in buf:
        left = end - time.time()
        if left <= 0:
            raise TimeoutError('timeout powershell')


        r, _, _ = select.select([fd], [], [], left)
        if r:
            chunk = os.read(fd, 4096)
            if not chunk:
                raise EOFError('powershell ferme')
            buf += chunk


    return buf.split(b'\n')[0].strip().decode()



def estimate_clock_offset(ps_cmd=None, trials=25):
    ps_cmd = ps_cmd or [
        'powershell.exe', '-NoProfile', '-NoLogo',
        '-NonInteractive', '-Command', '-'
    ]


    q = b"[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()\n"


    try:
        p = subprocess.Popen(
            ps_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )


        fd = p.stdout.fileno()
        best = None


        for k in range(trials + 3):
            t0 = time.time()
            p.stdin.write(q)
            p.stdin.flush()
            w = float(_readline_fd(fd, 20)) / 1000.0
            t1 = time.time()


            if k < 3:
                continue


            cand = (t1 - t0, w - (t0 + t1) / 2.0)
            if best is None or cand[0] < best[0]:
                best = cand


        try:
            p.stdin.close()
            p.terminate()
        except Exception:
            pass


        return best[1], best[0] / 2.0


    except Exception as e:
        print("[EB] ATTENTION : decalage d'horloge Windows/WSL non mesure (%s)" % e)
        return 0.0, NAN



class EnergiBridgeMonitor(object):
    COL = 'PACKAGE_ENERGY (J)'


    def __init__(self, eb_path, eb_outdir, name, interval_ms=200,
                 max_gap_s=5.0, wrap_j=262144.0, ps_cmd=None):


        self.eb_path = eb_path
        self.interval_ms = int(interval_ms)


        if self.interval_ms < 200:
            raise ValueError(
                "eb_interval doit etre >= 200 ms pour EnergiBridge "
                "(valeur recue : %d ms)." % self.interval_ms
            )


        self.max_gap_s = float(max_gap_s)
        self.wrap_j = float(wrap_j)


        eb_outdir = os.path.abspath(eb_outdir)
        os.makedirs(eb_outdir, exist_ok=True)


        self.name = name
        self.csv = os.path.join(
            eb_outdir,
            name + '_energibridge.csv'
        )


        self.ps_cmd = ps_cmd
        self.proc = None
        self.log = None
        self.offset = 0.0
        self.offset_unc = NAN
        self.glitches = 0


    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None


    def start(self):
        if not os.path.exists(self.eb_path):
            raise FileNotFoundError(self.eb_path)


        if os.path.exists(self.csv):
            os.remove(self.csv)


        win_csv = subprocess.check_output(
            ['wslpath', '-w', self.csv]
        ).decode().strip()


        self.offset, self.offset_unc = estimate_clock_offset(self.ps_cmd)


        print('[EB] Decalage horloge Windows-WSL : %.3f s (incertitude ~ %.3f s)'
              % (self.offset, self.offset_unc))


        cmd = [
            self.eb_path,
            '-o', win_csv,
            '-i', str(self.interval_ms),
            '--',
            'ping.exe', '-n', '2000000', '127.0.0.1'
        ]


        self.log = open(self.csv + '.log', 'w')


        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self.log
        )


        print('[EB] Enregistrement demarre :', self.csv)


        t_end = time.time() + 40


        while time.time() < t_end:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "EnergiBridge s'est arrete (code %s) ; voir %s.log"
                    % (self.proc.returncode, self.csv)
                )


            try:
                if len(self._read(self.COL)) >= 3:
                    break
            except Exception:
                pass


            time.sleep(0.5)
        else:
            raise RuntimeError('EnergiBridge ne produit aucun echantillon')


        time.sleep(2.0)


    def stop(self):
        if self.proc is not None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=8)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


            self.proc = None


        subprocess.call(
            ['taskkill.exe', '/F', '/IM', 'ping.exe'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


        if self.log is not None:
            try:
                self.log.close()
            except Exception:
                pass
            self.log = None


    def _read(self, col):
        wanted = ('Time', col)


        try:
            df = pd.read_csv(
                self.csv,
                usecols=lambda c: c in wanted,
                on_bad_lines='skip'
            )
        except TypeError:
            df = pd.read_csv(
                self.csv,
                usecols=lambda c: c in wanted,
                error_bad_lines=False
            )


        df = df.apply(pd.to_numeric, errors='coerce').dropna()


        if self.is_alive() and len(df) > 0:
            df = df.iloc[:-1]


        return df


    def _series(self, col):
        df = self._read(col)


        if col not in df.columns or len(df) < 3:
            return None


        t = df['Time'].to_numpy(float) / 1000.0 - self.offset
        e = df[col].to_numpy(float)


        o = np.argsort(t, kind='stable')
        t, e = t[o], e[o]


        k = np.concatenate(([True], np.diff(t) > 0))
        t, e = t[k], e[k]


        d = np.diff(e)


        big = (d < 0) & (np.abs(d) > self.wrap_j / 2.0)
        d[big] += self.wrap_j


        small = d < 0
        self.glitches += int(small.sum())
        d[small] = 0.0


        c = np.concatenate(([0.0], np.cumsum(d)))


        return t, c


    def wait_until(self, t_end, timeout=30.0):
        t_lim = time.time() + timeout


        while time.time() < t_lim:
            try:
                df = self._read(self.COL)


                if len(df) and df['Time'].iloc[-1] / 1000.0 - self.offset >= t_end + 0.05:
                    return True


            except Exception:
                pass


            time.sleep(0.5)


        return False


    def window(self, t0, t1, col=None):
        col = col or self.COL
        out = dict(E=NAN, n=0, dt=NAN, ok=False, why='')


        self.wait_until(t1)


        s = self._series(col)


        if s is None:
            out['why'] = 'moins de 3 echantillons'
            return out


        t, c = s
        out['dt'] = float(np.median(np.diff(t)))


        if t0 < t[0] or t1 > t[-1]:
            out['why'] = (
                'fenetre [%.1f, %.1f] s hors echantillons [0, %.1f] s'
                % (t0 - t[0], t1 - t[0], t[-1] - t[0])
            )
            return out


        i0 = int(np.searchsorted(t, t0, side='right'))
        i1 = int(np.searchsorted(t, t1, side='left'))


        gap0 = t[i0] - t[i0 - 1]
        gap1 = t[i1] - t[i1 - 1]


        out['n'] = int(((t >= t0) & (t <= t1)).sum())


        if max(gap0, gap1) > self.max_gap_s:
            out['why'] = (
                "trou d'echantillonnage de %.2f s a une borne (> %.1f s)"
                % (max(gap0, gap1), self.max_gap_s)
            )
            return out


        if out['n'] < 2:
            out['why'] = 'moins de 2 echantillons dans la fenetre'
            return out


        out['E'] = float(
            np.interp(t1, t, c) - np.interp(t0, t, c)
        )
        out['ok'] = True


        return out



# =====================================================================================
#  TEST
# =====================================================================================
def test(filedir, ckptdir_list, outdir, resultdir, scaling_factor=1.0,
         rho=1.0, res=1024, energy_source='none', eb_path=None,
         eb_outdir=None, eb_interval=200, eb_max_gap=5.0,
         eb_wrap_j=262144.0,
         pruning_mode='none', sparsity=0.0,
         simple_sparsity=0.25, complex_sparsity=0.0):


    start_time = time.time()
    x = load_sparse_tensor(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')


    if not os.path.exists(outdir):
        os.makedirs(outdir)


    if not os.path.exists(resultdir):
        os.makedirs(resultdir)


    filename = os.path.join(
        outdir,
        os.path.split(filedir)[-1].split('.')[0]
    )
    print('output filename:\t', filename)


    model = PCCModel().to(device)
    macs = MacCounter(model)


    base = os.path.split(filedir)[-1].split('.')[0]
    csv_name = os.path.join(resultdir, base + '.csv')


    all_results = None


    for idx, ckptdir in enumerate(ckptdir_list):
        print('=' * 10, idx + 1, '=' * 10)


        assert os.path.exists(ckptdir)
        ckpt = torch.load(ckptdir)
        model.load_state_dict(ckpt['model'])
        print('load checkpoint from \t', ckptdir)


        # --------------------------------------------------------------
        # PRUNING
        # --------------------------------------------------------------
        if pruning_mode == "static":
            model, pruning_report = apply_static_pruning(
                model,
                "pruning/config_static.json",
                sparsity
            )


            save_json(
                os.path.join(
                    resultdir,
                    "pruning_report_r%d.json" % (idx + 1)
                ),
                pruning_report
            )


            print("STATIC PRUNING APPLIED")
            print("Rate:", idx + 1)
            print("Sparsity:", sparsity)
            print("Active parameters:", pruning_report["active_parameters"])


        elif pruning_mode == "dynamic":
            model, pruning_report = apply_dynamic_pruning(
                model,
                "pruning/config_dynamic.json",
                x.C[:, 1:].detach().cpu(),
                simple_sparsity,
                complex_sparsity
            )


            save_json(
                os.path.join(
                    resultdir,
                    "pruning_report_r%d.json" % (idx + 1)
                ),
                pruning_report
            )


            print("DYNAMIC PRUNING APPLIED")
            print("Rate:", idx + 1)
            print("Simple tiles:", pruning_report["num_simple_tiles"])
            print("Complex tiles:", pruning_report["num_complex_tiles"])
            print("Effective sparsity:", pruning_report["effective_sparsity"])


        # --------------------------------------------------------------
        # Parametres apres pruning
        # --------------------------------------------------------------
        n_params = int(sum(p.numel() for p in model.parameters()))
        n_nonzero = int(sum((p != 0).sum().item() for p in model.parameters()))


        coder = Coder(model=model, filename=filename)
        postfix_idx = '_r' + str(idx + 1)


        if scaling_factor != 1:
            x_in = scale_sparse_tensor(x, factor=scaling_factor)
        else:
            x_in = x


        mon = None


        if energy_source == 'energibridge':
            mon = EnergiBridgeMonitor(
                eb_path,
                eb_outdir or resultdir,
                "%s_r%d" % (base, idx + 1),
                eb_interval,
                eb_max_gap,
                eb_wrap_j
            )
            mon.start()


        # --------------------------------------------------------------
        # Encodage
        # --------------------------------------------------------------
        macs.reset()


        with MemProbe() as mem_enc:
            t_enc0 = time.time()
            start_time = time.time()
            _ = coder.encode(x_in, postfix=postfix_idx)
            time_enc = round(time.time() - start_time, 3)
            t_enc1 = time.time()


        print('Enc Time:\t', time_enc, 's')
        macs_enc = macs.compute()


        # --------------------------------------------------------------
        # Decodage
        # --------------------------------------------------------------
        macs.reset()


        with MemProbe() as mem_dec:
            t_dec0 = time.time()
            start_time = time.time()
            x_dec = coder.decode(postfix=postfix_idx, rho=rho)
            time_dec = round(time.time() - start_time, 3)
            t_dec1 = time.time()


        print('Dec Time:\t', time_dec, 's')
        macs_dec = macs.compute()


        # --------------------------------------------------------------
        # Post-traitement
        # --------------------------------------------------------------
        if scaling_factor != 1:
            x_dec = scale_sparse_tensor(x_dec, factor=1.0 / scaling_factor)


        bits = np.array([
            os.path.getsize(filename + postfix_idx + postfix) * 8
            for postfix in [
                '_C.bin', '_F.bin', '_H.bin', '_num_points.bin'
            ]
        ])


        bpps = (bits / len(x)).round(3)
        print('bits:\t', sum(bits))
        print('bpps:\t', sum(bpps).round(3))


        start_time = time.time()
        write_ply_ascii_geo(
            filename + postfix_idx + '_dec.ply',
            x_dec.C.detach().cpu().numpy()[:, 1:]
        )
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')


        start_time = time.time()
        pc_error_metrics = pc_error(
            filedir,
            filename + postfix_idx + '_dec.ply',
            res=res,
            normal=True,
            show=False
        )
        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])


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
        results["latency(codec)[s]"] = time_enc + time_dec


        # --------------------------------------------------------------
        # MACs, parametres et memoire
        # --------------------------------------------------------------
        results["MACs(enc)"] = macs_enc
        results["MACs(dec)"] = macs_dec
        results["MACs(codec)"] = macs_enc + macs_dec
        results["params"] = n_params
        results["params_nonzero"] = n_nonzero


        results["mem_peak(codec)[MB]"] = max(
            mem_enc.peak_total_mb,
            mem_dec.peak_total_mb
        )


        # --------------------------------------------------------------
        # Energie : fenetres encodage / decodage
        # --------------------------------------------------------------
        e_enc = e_dec = NAN
        valid = False


        if mon is not None:
            we = mon.window(t_enc0, t_enc1)
            wd = mon.window(t_dec0, t_dec1)


            valid = bool(we['ok'] and wd['ok'])
            e_enc, e_dec = we['E'], wd['E']


            if not we['ok']:
                print('[EB] Fenetre encodage INVALIDE :', we['why'])
            if not wd['ok']:
                print('[EB] Fenetre decodage INVALIDE :', wd['why'])


            mon.stop()
            mon = None


        results["energy(enc)[J]"] = e_enc
        results["energy(dec)[J]"] = e_dec
        results["energy(codec)[J]"] = e_enc + e_dec


        results["power_avg(enc)[W]"] = (
            e_enc / (t_enc1 - t_enc0) if valid else NAN
        )
        results["power_avg(dec)[W]"] = (
            e_dec / (t_dec1 - t_dec0) if valid else NAN
        )


        # --------------------------------------------------------------
        # Informations de pruning dans le CSV
        # --------------------------------------------------------------
        results["pruning_mode"] = pruning_mode
        results["sparsity"] = sparsity
        results["simple_sparsity"] = simple_sparsity
        results["complex_sparsity"] = complex_sparsity


        if pruning_mode == "static":
            results["effective_sparsity"] = sparsity


        elif pruning_mode == "dynamic":
            results["effective_sparsity"] = pruning_report["effective_sparsity"]
            results["num_simple_tiles"] = pruning_report["num_simple_tiles"]
            results["num_complex_tiles"] = pruning_report["num_complex_tiles"]


        else:
            results["effective_sparsity"] = 0.0


        if idx == 0:
            all_results = results.copy(deep=True)
        else:
            all_results = pd.concat(
                [all_results, results],
                ignore_index=True
            )


        all_results.to_csv(csv_name, index=False)
        print('Wrile results to: \t', csv_name)


        print(
            'Rate %d | Enc: %.3f J | Dec: %.3f J | Codec: %.3f J | '
            'Mem pic: %.1f MB | MACs codec: %.4g' % (
                idx + 1,
                e_enc,
                e_dec,
                e_enc + e_dec,
                all_results["mem_peak(codec)[MB]"].iloc[-1],
                macs_enc + macs_dec
            )
        )


    return all_results



if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument("--filedir",
                        default='../../../testdata/8iVFB/longdress_vox10_1300.ply')
    parser.add_argument("--outdir", default='./output')
    parser.add_argument("--resultdir", default='./results')
    parser.add_argument("--scaling_factor", type=float, default=1.0)
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--energy_source", default='none',
                        choices=['none', 'energibridge'])
    parser.add_argument("--eb_path", default=None)
    parser.add_argument("--eb_outdir", default=None)
    parser.add_argument("--eb_interval", type=int, default=200,
                        help="intervalle EnergiBridge en ms ; doit etre >= 200")
    parser.add_argument("--eb_max_gap", type=float, default=5.0)
    parser.add_argument("--eb_wrap_j", type=float, default=262144.0)
    parser.add_argument("--max_rates", type=int, default=7)


    parser.add_argument(
        "--pruning_mode",
        type=str,
        default="none",
        choices=["none", "static", "dynamic"],
        help="Type de pruning"
    )


    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
        help="Taux de pruning statique : 0.10, 0.25, 0.50"
    )


    parser.add_argument(
        "--simple_sparsity",
        type=float,
        default=0.25,
        help="Pruning des tuiles simples"
    )


    parser.add_argument(
        "--complex_sparsity",
        type=float,
        default=0.0,
        help="Pruning des tuiles complexes"
    )


    args = parser.parse_args()


    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir):
        os.makedirs(args.resultdir)


    ckptdir_list = [
        './ckpts/r1_0.025bpp.pth',
        './ckpts/r2_0.05bpp.pth',
        './ckpts/r3_0.10bpp.pth',
        './ckpts/r4_0.15bpp.pth',
        './ckpts/r5_0.25bpp.pth',
        './ckpts/r6_0.3bpp.pth',
        './ckpts/r7_0.4bpp.pth'
    ][:args.max_rates]


    all_results = test(
        args.filedir,
        ckptdir_list,
        args.outdir,
        args.resultdir,
        scaling_factor=args.scaling_factor,
        rho=args.rho,
        res=args.res,
        energy_source=args.energy_source,
        eb_path=args.eb_path,
        eb_outdir=args.eb_outdir,
        eb_interval=args.eb_interval,
        eb_max_gap=args.eb_max_gap,
        eb_wrap_j=args.eb_wrap_j,
        pruning_mode=args.pruning_mode,
        sparsity=args.sparsity,
        simple_sparsity=args.simple_sparsity,
        complex_sparsity=args.complex_sparsity
    )


    import matplotlib.pyplot as plt


    filename = os.path.split(args.filedir)[-1][:-4]


    # --------------------------------------------------------------
    # Figure unique : courbe RD (bpp vs D1/D2)
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))


    ax.plot(
        np.array(all_results["bpp"][:]),
        np.array(all_results["mseF,PSNR (p2point)"][:]),
        label="D1", marker='x', color='red'
    )


    ax.plot(
        np.array(all_results["bpp"][:]),
        np.array(all_results["mseF,PSNR (p2plane)"][:]),
        label="D2", marker='x', color='blue'
    )


    ax.set_title(filename)
    ax.set_xlabel('bpp')
    ax.set_ylabel('PSNR (dB)')
    ax.grid(ls='-.')
    ax.legend(loc='lower right')

#################final ########################################
    fig.savefig(os.path.join(args.resultdir, filename + '.jpg'))