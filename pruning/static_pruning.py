from pruning.pruning_utils import (
    load_config,
    get_conv_modules,
    get_weight_tensor,
    apply_mask_to_module,
    count_active_parameters,
    count_pruned_output_channels,
    save_json
)


def apply_static_pruning(model, config_path, sparsity):
    """
    Pruning statique global.

    Le même taux de sparsité est appliqué à toutes les convolutions
    MinkowskiEngine du modèle.

    Les poids des groupes de canaux les moins importants sont mis à zéro.
    Les MACs ne diminuent pas automatiquement, car il s'agit d'un
    pruning par masque et non d'une suppression structurelle.
    """

    config = load_config(config_path)
    group_size = config["group_size"]

    conv_layers = get_conv_modules(model)

    if len(conv_layers) == 0:
        raise RuntimeError(
            "Aucune couche MinkowskiConvolution detectee. "
            "Le pruning ne peut pas etre applique."
        )

    report = {
        "mode": "static",
        "sparsity": sparsity,
        "group_size": group_size,
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
            sparsity
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