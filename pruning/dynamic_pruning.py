import torch

from pruning.pruning_utils import (
    load_config,
    get_conv_modules,
    get_weight_tensor,
    apply_mask_to_module,
    count_active_parameters,
    count_pruned_output_channels,
    save_json
)


def compute_tile_density(points, tile_size):
    """
    Calcule la densité locale de chaque tuile 3D.

    points : tenseur [N, 3] contenant les coordonnées des points.
    tile_size : taille de tuile en unités voxel.
    """

    min_coords = points.min(dim=0).values
    max_coords = points.max(dim=0).values

    num_tiles = (
        ((max_coords - min_coords) / tile_size)
        .ceil()
        .long()
        .clamp(min=1)
    )

    tile_ids = ((points - min_coords) / tile_size).long()

    tile_ids[:, 0] = tile_ids[:, 0].clamp(0, num_tiles[0] - 1)
    tile_ids[:, 1] = tile_ids[:, 1].clamp(0, num_tiles[1] - 1)
    tile_ids[:, 2] = tile_ids[:, 2].clamp(0, num_tiles[2] - 1)

    unique_tiles, counts = torch.unique(
        tile_ids,
        dim=0,
        return_counts=True
    )

    tile_volume = tile_size ** 3

    tile_info = {}

    for i in range(unique_tiles.shape[0]):
        key = "_".join(
            str(int(x)) for x in unique_tiles[i].tolist()
        )

        density = counts[i].item() / tile_volume

        tile_info[key] = {
            "num_points": int(counts[i].item()),
            "density": float(density)
        }

    return tile_info


def classify_tiles(tile_info, density_threshold):
    """
    Classe chaque tuile comme simple ou complexe.

    simple : densité locale inférieure au seuil
    complex : densité locale supérieure ou égale au seuil
    """

    classification = {}

    for tile_id, info in tile_info.items():
        if info["density"] < density_threshold:
            classification[tile_id] = "simple"
        else:
            classification[tile_id] = "complex"

    return classification


def apply_dynamic_pruning(
    model,
    config_path,
    points,
    simple_sparsity,
    complex_sparsity
):
    """
    Pruning dynamique à deux niveaux.

    Les tuiles peu denses activent `simple_sparsity`.
    Les tuiles denses activent `complex_sparsity`.

    La sparsité effective est une moyenne pondérée par la proportion
    de tuiles simples et complexes.

    Remarque importante :
    les masques sont réellement appliqués aux convolutions
    MinkowskiEngine, mais le masque final reste global pour une passe
    complète d'encodage et de décodage.
    """

    config = load_config(config_path)

    tile_size = config["tile_size"]
    density_threshold = config["density_threshold"]
    group_size = config["group_size"]

    points = points.detach().cpu().float()

    tile_info = compute_tile_density(
        points,
        tile_size
    )

    tile_classification = classify_tiles(
        tile_info,
        density_threshold
    )

    num_simple = 0
    num_complex = 0

    for tile_class in tile_classification.values():
        if tile_class == "simple":
            num_simple += 1
        else:
            num_complex += 1

    total_tiles = num_simple + num_complex

    simple_ratio = num_simple / max(1, total_tiles)
    complex_ratio = num_complex / max(1, total_tiles)

    effective_sparsity = (
        simple_ratio * simple_sparsity
        + complex_ratio * complex_sparsity
    )

    conv_layers = get_conv_modules(model)

    if len(conv_layers) == 0:
        raise RuntimeError(
            "Aucune couche MinkowskiConvolution detectee. "
            "Le pruning dynamique ne peut pas etre applique."
        )

    report = {
        "mode": "dynamic",
        "tile_size": tile_size,
        "density_threshold": density_threshold,
        "group_size": group_size,
        "simple_sparsity": simple_sparsity,
        "complex_sparsity": complex_sparsity,
        "effective_sparsity": effective_sparsity,
        "num_tiles": total_tiles,
        "num_simple_tiles": num_simple,
        "num_complex_tiles": num_complex,
        "simple_tile_ratio": simple_ratio,
        "complex_tile_ratio": complex_ratio,
        "num_conv_layers": len(conv_layers),
        "layers": {},
        "skipped_layers": []
    }

    for name, module in conv_layers:
        out_channels = get_weight_tensor(module).shape[0]

        if out_channels < group_size:
            report["skipped_layers"].append(name)

            report["layers"][name] = {
                "pruned_groups": 0,
                "group_size": group_size,
                "pruned_output_channels": 0,
                "out_channels": out_channels,
                "skipped": True
            }

            continue

        pruned_groups = apply_mask_to_module(
            module,
            group_size,
            effective_sparsity
        )

        report["layers"][name] = {
            "pruned_groups": int(pruned_groups),
            "group_size": group_size,
            "pruned_output_channels": int(
                getattr(module, "_pruned_out_channels", 0)
            ),
            "out_channels": out_channels,
            "skipped": False
        }

    total, active = count_active_parameters(model)
    pruned_channels = count_pruned_output_channels(model)

    report["total_parameters"] = total
    report["active_parameters"] = active
    report["parameter_sparsity"] = 1.0 - active / total
    report["pruned_output_channels"] = pruned_channels

    return model, report