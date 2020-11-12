import torch.nn as nn
import torch.nn.functional as F
from monai.losses.dice import DiceLoss, GeneralizedDiceLoss


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
        if target.ndim == 3:  # Shape: (N, H, W)
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
        if target.ndim == 3:  # Shape: (N, H, W)
            target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)
        return (input, target)


class DiceMetricWrapper(object):
    """TODO"""

    def __init__(self):
        self.metric_fx = DiceLoss(
            include_background=False, to_onehot_y=True, softmax=True, reduction="none"
        )

    def __call__(self, input, target):
        input, target = self._process(input, target)
        score = self.metric_fx(input, target)  # Shape: (N, C)
        return 1 - score.mean(dim=0)

    def _process(self, input, target):
        if target.ndim == 3:  # Shape: (N, H, W)
            target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)
        return (input, target)


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


def _batch_masked_mean(array, mask):
    """
    array, mask - Shape: (N, C)
    """
    masked_values = array * mask
    if mask.shape[-1] == 1:
        return masked_values.mean()
    return (masked_values.sum(axis=-1) / mask.sum(axis=-1)).mean()
