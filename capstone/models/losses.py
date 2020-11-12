from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from capstone.models.metrics import compute_meandice, do_metric_reduction
from monai.losses.dice import DiceLoss, GeneralizedDiceLoss
from monai.transforms import AsDiscrete


class BaseLossWrapper(nn.Module):
    """TODO"""

    def __init__(self) -> None:
        super(BaseLossWrapper, self).__init__()

    @property
    def loss_fx(self):
        raise NotImplementedError

    def forward(self, input, target):
        input, target = self._process(input, target)
        return self.loss_fx(input, target)

    def _process(self, input, target):
        return (input, target)


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

    def _process(self, input, target):
        assert target.ndim == 3, "Expected target of shape: (N, H, W)"
        target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)

        return (input, target)


class GeneralizedDiceLossWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self):
        super(GeneralizedDiceLossWrapper, self).__init__()

    @property
    def loss_fx(self):
        return GeneralizedDiceLoss(
            include_background=False, to_onehot_y=True, softmax=True
        )

    def _process(self, input, target):
        assert target.ndim == 3, "Expected target of shape: (N, H, W)"
        target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)

        return (input, target)


class DiceMetricWrapper(object):
    """TODO"""

    def __init__(self):
        self.metric_fx = partial(compute_meandice, include_background=False)

    def __call__(self, input, target):
        input, target = self._process(input, target)
        score = self.metric_fx(input, target)  # Shape: (N, C)

        dice_per_class = do_metric_reduction(score, "mean_batch")[0]
        dice_mean = do_metric_reduction(score, "mean")[0]
        return dice_mean, dice_per_class

    def _process(self, input, target):
        assert input.ndim == 4, "Expected input of shape: (N, C, H, W)"
        assert target.ndim == 3, "Expected target of shape: (N, H, W)"
        n_classes = input.shape[1]

        # The line below works from PyTorch 1.7 onwards. Will fail for previous
        # versions due to changes in `max()` and `argmax()`
        input = (
            torch.softmax(input, dim=1).argmax(dim=1).unsqueeze(dim=1)
        )  # Shape: (N, 1, H, W)
        target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)

        expand = AsDiscrete(to_onehot=True, n_classes=n_classes)
        return expand(input), expand(target)  # Binarized


LOSSES = {
    "CrossEntropy": CrossEntropyWrapper(),
    "Dice": DiceLossWrapper(),
    "GeneralizedDice": GeneralizedDiceLossWrapper(),
}


class MultipleLossWrapper(nn.Module):
    def __init__(self, losses):
        super(MultipleLossWrapper, self).__init__()
        for name in losses:
            assert name in LOSSES.keys()
        self.losses = nn.ModuleDict({name: LOSSES[name] for name in losses})

    def forward(self, input, target):
        values = {name: fx(input, target) for (name, fx) in self.losses.items()}
        return values


# def _batch_masked_mean(array, mask):
#     """
#     array, mask - Shape: (N, C)
#     """
#     masked_values = array * mask
#     if mask.shape[-1] == 1:
#         return masked_values.mean()
#     return (masked_values.sum(axis=-1) / mask.sum(axis=-1)).mean()
