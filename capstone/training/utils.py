"""
The functions `_squash_masks()` and `_squash_predictions()` work from PyTorch 1.7
onwards. Will fail for previous versions due to changes in `max()` and `argmax()`
"""

import numpy as np
import torch

RNG = np.random.default_rng(seed=12342)


def _squash_masks(masks, n_classes, device):
    _temp = torch.arange(1, n_classes, device=device)
    masks = (masks * _temp[None, :, None, None]).max(dim=1).values
    return masks


def _squash_predictions(preds):
    return torch.softmax(preds, dim=1).argmax(dim=1)  # Shape: (N, H, W)


def mixup_data(images, alpha=0.2, device=None):
    """Code adapted from: https://github.com/facebookresearch/mixup-cifar10"""
    batch_size = images.shape[0]

    lambda_ = RNG.beta(alpha, alpha)
    index = torch.randperm(batch_size, device=device)
    mixed_images = mixup_tensors(images, images[index], lambda_)

    return mixed_images, index, lambda_


def mixup_tensors(tensor_1, tensor_2, lambda_):
    return lambda_ * tensor_1 + (1 - lambda_) * tensor_2
