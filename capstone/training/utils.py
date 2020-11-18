"""
These functions works from PyTorch 1.7 onwards. Will fail for previous
versions due to changes in `max()` and `argmax()`
"""

import torch


def _squash_masks(masks, n_classes, device):
    _temp = torch.arange(1, n_classes, device=device)
    masks = (masks * _temp[None, :, None, None]).max(dim=1).values
    return masks


def _squash_predictions(preds):
    return torch.softmax(preds, dim=1).argmax(dim=1)  # Shape: (N, H, W)
