import os
import json

import torch
import MinkowskiEngine as ME


ME_CONV_TYPES = (
    ME.MinkowskiConvolution,
    ME.MinkowskiConvolutionTranspose,
    ME.MinkowskiGenerativeConvolutionTranspose,
)


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def get_conv_modules(model):
    """
    Retourne uniquement les couches de convolution sparse
    MinkowskiEngine réellement utilisées par PCGCv2.
    """

    modules = []

    for name, module in model.named_modules():
        if isinstance(module, ME_CONV_TYPES):
            modules.append((name, module))

    return modules


def get_weight_tensor(module):
    """
    MinkowskiEngine stocke les poids dans `kernel`,
    contrairement aux convolutions PyTorch classiques (`weight`).
    """

    if hasattr(module, "kernel"):
        return module.kernel

    if hasattr(module, "weight"):
        return module.weight

    raise AttributeError(
        "Le module %s n'a ni `kernel` ni `weight`."
        % type(module).__name__
    )


def compute_group_scores(module, group_size):
    """
    Calcule le score L1 de chaque groupe de canaux de sortie.

    Un groupe correspond à `group_size` canaux consécutifs.

    Si une couche possède moins de `group_size` canaux de sortie,
    aucun groupe complet ne peut être formé : la fonction retourne
    un tenseur vide.
    """

    weight = get_weight_tensor(module).detach()
    out_channels = weight.shape[0]

    if out_channels < group_size:
        return torch.tensor([], dtype=weight.dtype)

    usable = (out_channels // group_size) * group_size

    if usable == 0:
        return torch.tensor([], dtype=weight.dtype)

    weight = weight[:usable]

    grouped = weight.reshape(
        weight.shape[0] // group_size,
        group_size,
        -1
    )

    scores = grouped.abs().sum(dim=(1, 2))

    return scores


def select_groups_to_prune(module, group_size, sparsity):
    """
    Sélectionne les groupes de canaux ayant les scores L1 les plus faibles.

    Retourne un tenseur vide si aucun groupe ne peut être pruné.
    """

    scores = compute_group_scores(module, group_size)

    if scores.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    num_groups = scores.shape[0]
    num_prune = int(round(num_groups * sparsity))

    if num_prune == 0:
        return torch.tensor([], dtype=torch.long)

    _, indices = torch.topk(
        scores,
        k=num_prune,
        largest=False
    )

    return indices.sort().values


def apply_mask_to_module(module, group_size, sparsity):
    """
    Applique réellement un masque de pruning structuré par groupes
    de canaux de sortie sur une convolution MinkowskiEngine.

    Les couches dont le nombre de canaux de sortie est inférieur
    à `group_size` ne sont pas prunées.
    """

    weight = get_weight_tensor(module)
    out_channels = weight.shape[0]

    if out_channels < group_size:
        module._pruned_out_channels = 0
        return 0

    indices = select_groups_to_prune(
        module,
        group_size,
        sparsity
    )

    if len(indices) == 0:
        module._pruned_out_channels = 0
        return 0

    with torch.no_grad():
        for group_id in indices.tolist():
            start = group_id * group_size
            end = start + group_size

            if end <= out_channels:
                weight[start:end] = 0.0

        if getattr(module, "bias", None) is not None:
            bias = module.bias

            for group_id in indices.tolist():
                start = group_id * group_size
                end = start + group_size

                if end <= bias.shape[0]:
                    bias[start:end] = 0.0

    module._pruned_out_channels = int(len(indices) * group_size)

    return len(indices)


def count_active_parameters(model):
    """
    Compte les paramètres totaux et non nuls du modèle.
    """

    total = 0
    active = 0

    for param in model.parameters():
        total += param.numel()
        active += torch.count_nonzero(param).item()

    return total, active


def count_pruned_output_channels(model):
    """
    Retourne le nombre total de canaux de sortie mis à zéro.

    Cette valeur ne modifie pas les MACs réellement exécutés :
    elle indique seulement le potentiel de pruning structurel.
    """

    total_pruned = 0

    for _, module in get_conv_modules(model):
        total_pruned += getattr(
            module,
            "_pruned_out_channels",
            0
        )

    return total_pruned


def save_json(path, data):
    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)