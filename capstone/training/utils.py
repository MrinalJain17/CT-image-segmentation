"""
The functions `_squash_masks()` and `_squash_predictions()` work from PyTorch 1.7
onwards. Will fail for previous versions due to changes in `max()` and `argmax()`
"""

import numpy as np
import torch

RNG = np.random.default_rng(seed=12342)
ANNOTATION_COUNT = torch.as_tensor([601, 44, 601, 94, 88, 535, 549, 280, 253])


def _squash_masks(masks, n_classes, device):
    _temp = torch.arange(1, n_classes, device=device)
    masks = (masks * _temp[None, :, None, None]).max(dim=1).values
    return masks


def _squash_predictions(preds):
    return torch.softmax(preds, dim=1).argmax(dim=1)  # Shape: (N, H, W)


def weighted_mixup(images, masks, alpha=0.2, device=None):
    batch_size = images.shape[0]

    count = ANNOTATION_COUNT.type_as(images)
    structure_indicator = ((masks == 1).sum(dim=(2, 3)) > 0).float()  # Shape: (N, C)
    structure_indicator = torch.einsum("ij,j->ij", structure_indicator, count)

    probability = 1.0 / (
        structure_indicator.sum(dim=1) / (structure_indicator > 0).sum(dim=1)
    )
    probability = probability / probability.sum()  # Normalizing to sum to 1

    lambda_ = RNG.beta(alpha, alpha)
    index = torch.from_numpy(
        RNG.choice(
            batch_size, batch_size, replace=True, p=probability.detach().cpu().numpy()
        )
    ).to(device)
    mixed_images = mixup_tensors(images, images[index], lambda_)

    return mixed_images, index, lambda_


def mixup_data(images, alpha=0.2, device=None):
    batch_size = images.shape[0]

    lambda_ = RNG.beta(alpha, alpha)
    index = torch.randperm(batch_size, device=device)
    mixed_images = mixup_tensors(images, images[index], lambda_)

    return mixed_images, index, lambda_


def mixup_tensors(tensor_1, tensor_2, lambda_):
    return lambda_ * tensor_1 + (1 - lambda_) * tensor_2
