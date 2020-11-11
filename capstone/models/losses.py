from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from monai.losses.dice import DiceLoss, GeneralizedDiceLoss
from pytorch_lightning.metrics.functional import dice_score


class BaseLossWrapper(nn.Module):
    """TODO"""

    def __init__(self) -> None:
        super(BaseLossWrapper, self).__init__()

    @property
    def loss_fx(self):
        raise NotImplementedError

    def forward(self, input, target):
        return self.loss_fx(input, target)


class CrossEntropyWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self):
        super(CrossEntropyWrapper, self).__init__()

    @property
    def loss_fx(self):
        return F.cross_entropy


class DiceLossWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self):
        super(DiceLossWrapper, self).__init__()

    @property
    def loss_fx(self):
        return DiceLoss(include_background=False, to_onehot_y=True, softmax=True)


class GeneralizedDiceLossWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self):
        super(GeneralizedDiceLossWrapper, self).__init__()

    @property
    def loss_fx(self):
        return GeneralizedDiceLoss(
            include_background=False, to_onehot_y=True, softmax=True
        )


class DiceMetricWrapper(object):
    """TODO"""

    def __init__(self):
        self.loss_fx = partial(dice_score, bg=False)

    def __call__(self, input, target):
        return self.loss_fx(input, target)


def _batch_masked_mean(array, mask):
    """
    array, mask - Shape: (N, C)
    """
    masked_values = array * mask
    if mask.shape[-1] == 1:
        return masked_values.mean()
    return (masked_values.sum(axis=-1) / mask.sum(axis=-1)).mean()
