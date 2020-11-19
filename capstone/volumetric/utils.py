import torch


def _squash_masks_3D(masks, n_classes, device):
    _temp = torch.arange(1, n_classes, device=device)
    masks = (masks * _temp[None, :, None, None, None]).max(dim=1).values
    return masks
