import torch
import torch.nn.functional as F


def binary_cross_entropy_with_logits(input, target, mask_indicator, **kwargs):
    loss_per_class = F.binary_cross_entropy_with_logits(
        input, target, reduction="none", **kwargs
    ).mean(
        dim=(2, 3)
    )  # Shape: (N, C) where C is the number of segmentation classes

    annotations_per_image = (
        mask_indicator.sum(dim=1) if mask_indicator.shape[1] != 1 else None
    )
    effective_loss = torch.einsum("nc,nc->n", loss_per_class, mask_indicator)
    if annotations_per_image is not None:  # Multiple classes
        effective_loss /= annotations_per_image
    return effective_loss.mean()
