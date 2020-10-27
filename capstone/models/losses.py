from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from monai.losses.dice import DiceLoss


class BaseLossWrapper(nn.Module):
    def __init__(self, apply_mask=True):
        super(BaseLossWrapper, self).__init__()
        self.loss_fx = None  # Required to be implemented in the subclasses
        self.apply_mask = apply_mask

    def _compute_loss_per_mask(self, input, target):
        raise NotImplementedError()

    def forward(self, input, target, mask_indicator=None):
        """
        input, target - Shape: (N, C, H, W)
        mask_indicator - Shape: (N, C)
        """
        loss_per_mask = self._compute_loss_per_mask(input, target)  # Shape: (N, C)

        if self.apply_mask:
            assert mask_indicator is not None, "Mask indicators not provided"
            return _batch_masked_mean(loss_per_mask, mask_indicator)  # Scalar

        return loss_per_mask.mean()  # Scalar


class BCELossWrapper(BaseLossWrapper):
    def __init__(self, apply_mask=True, **kwargs):
        super(BCELossWrapper, self).__init__(apply_mask=apply_mask)
        self.loss_fx = partial(
            F.binary_cross_entropy_with_logits, reduction="none", **kwargs
        )

    def _compute_loss_per_mask(self, input, target):
        return self.loss_fx(input, target).mean(dim=(2, 3))  # Shape: (N, C)


class DiceLossWrapper(BaseLossWrapper):
    def __init__(self, apply_mask=True):
        super(DiceLossWrapper, self).__init__(apply_mask=apply_mask)
        self.loss_fx = DiceLoss(sigmoid=True, reduction="none")

    def _compute_loss_per_mask(self, input, target):
        return self.loss_fx(input, target)  # Shape: (N, C)


def _batch_masked_mean(array, mask):
    """
    array, mask - Shape: (N, C)
    """
    masked_values = array * mask
    if mask.shape[-1] == 1:
        return masked_values.mean()
    return (masked_values.sum(axis=-1) / mask.sum(axis=-1)).mean()
