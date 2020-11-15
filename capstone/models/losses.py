import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.dice import DiceLoss, GeneralizedDiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.losses.tversky import TverskyLoss

WEIGHT = {
    "Background": 1e-10,
    "BrainStem": 0.007,
    "Chiasm": 0.3296,
    "Mandible": 0.0046,
    "OpticNerve_L": 0.2619,
    "OpticNerve_R": 0.3035,
    "Parotid_L": 0.0068,
    "Parotid_R": 0.0065,
    "Submandibular_L": 0.0374,
    "Submandibular_R": 0.0426,
}  # Inverse pixel-frequency (Background is given no weight)


class BaseLossWrapper(nn.Module):
    """TODO"""

    def __init__(self):
        super(BaseLossWrapper, self).__init__()

    @property
    def loss_fx(self):
        raise NotImplementedError

    def forward(self, input, target):
        input, target = self._process(input, target)
        return self.loss_fx(input, target)

    def _process(self, input, target):
        assert target.ndim == 3, "Expected target of shape: (N, H, W)"
        target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)

        return (input, target)


class CrossEntropyWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self):
        super(CrossEntropyWrapper, self).__init__()

    @property
    def loss_fx(self):
        return F.cross_entropy

    def _process(self, input, target):
        return (input, target)


class WeightedCrossEntropyWrapper(CrossEntropyWrapper):
    """TODO"""

    def __init__(self):
        super(WeightedCrossEntropyWrapper, self).__init__()
        self.weight = torch.as_tensor(list(WEIGHT.values()))

    def forward(self, input, target):
        input, target = self._process(input, target)
        return self.loss_fx(input, target, weight=self.weight.type_as(input))


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


class FocalLossWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self):
        super(FocalLossWrapper, self).__init__()

    @property
    def loss_fx(self):
        return FocalLoss()


class TverskyLossWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self):
        super(TverskyLossWrapper, self).__init__()

    @property
    def loss_fx(self):
        return TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            alpha=0.25,  # Weight of false positives (precision)
            beta=0.75,  # Weight of false negatives  (recall)
        )


LOSSES = {
    "CrossEntropy": CrossEntropyWrapper(),
    "Dice": DiceLossWrapper(),
    "GeneralizedDice": GeneralizedDiceLossWrapper(),
    "WeightedCrossEntropy": WeightedCrossEntropyWrapper(),
    "Focal": FocalLossWrapper(),
    "Tversky": TverskyLossWrapper(),
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
