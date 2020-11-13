from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from capstone.models.metrics import compute_meandice, do_metric_reduction
from capstone.utils import miccai
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
        self.n_classes = len(miccai.STRUCTURES) + 1  # Additional background

    def __call__(self, input, target):
        input, target = self._process(input, target)
        score = self.metric_fx(input, target)  # Shape: (N, C)

        dice_per_class = do_metric_reduction(score, "mean_batch")[0]
        dice_mean = dice_per_class.mean()
        return dice_mean, dice_per_class

    def _process(self, input, target):
        assert input.ndim == 3, "Expected input of shape: (N, H, W)"
        assert target.ndim == 3, "Expected target of shape: (N, H, W)"

        input = input.unsqueeze(dim=1)  # Shape: (N, 1, H, W)
        target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)

        expand = AsDiscrete(to_onehot=True, n_classes=self.n_classes)
        return expand(input), expand(target)  # Shape: (N, C, H, W) - binarized


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
