import torch


def _squash_masks(self, masks, n_classes, device):
    _temp = torch.arange(1, n_classes, device)
    masks = (masks * _temp[None, :, None, None]).max(dim=1).values
    return masks
