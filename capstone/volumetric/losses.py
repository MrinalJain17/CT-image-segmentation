from capstone.models.losses import BaseLossWrapper, MultipleLossWrapper
from capstone.models.temp import GeneralizedDiceLoss
from capstone.utils import miccai
from monai.losses.dice import DiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.transforms import AsDiscrete
import torch
import torch.nn.functional as F

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


class BaseLossWrapper3D(BaseLossWrapper):
    """TODO"""

    def __init__(self):
        super(BaseLossWrapper3D, self).__init__()

    def _process(self, input, target):
        # assert target.ndim == 3, "Expected target of shape: (N, H, W)"
        target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)

        return (input, target)


class CrossEntropyWrapper3D(BaseLossWrapper3D):
    """TODO"""

    def __init__(self, **kwargs):
        super(CrossEntropyWrapper3D, self).__init__()

    @property
    def loss_fx(self):
        return F.cross_entropy

    def _process(self, input, target):
        return (input, target)


class WeightedCrossEntropyWrapper3D(CrossEntropyWrapper3D):
    """TODO"""

    def __init__(self, **kwargs):
        super(WeightedCrossEntropyWrapper3D, self).__init__()
        self.weight = torch.as_tensor(list(WEIGHT.values()))

    def forward(self, input, target):
        input, target = self._process(input, target)
        return self.loss_fx(input, target, weight=self.weight.type_as(input))


class DiceLossWrapper3D(BaseLossWrapper3D):
    """TODO"""

    def __init__(self, reduction="mean"):
        super(DiceLossWrapper3D, self).__init__()
        self.reduction = reduction

    @property
    def loss_fx(self):
        return DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            reduction=self.reduction,
        )


class GeneralizedDiceLossWrapper3D(BaseLossWrapper3D):
    """TODO"""

    def __init__(self, reduction="mean"):
        super(GeneralizedDiceLossWrapper3D, self).__init__()
        self.reduction = reduction

    @property
    def loss_fx(self):
        return GeneralizedDiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            reduction=self.reduction,
        )


class FocalLossWrapper3D(BaseLossWrapper3D):
    """TODO"""

    def __init__(self, reduction="mean"):
        super(FocalLossWrapper3D, self).__init__()
        self.reduction = reduction
        self.n_classes = len(miccai.STRUCTURES) + 1  # Additional background

    @property
    def loss_fx(self):
        return FocalLoss(reduction=self.reduction)

    def _process(self, input, target):
        # assert input.ndim == 4, "Expected input of shape: (N, C, H, W)"
        # assert target.ndim == 3, "Expected target of shape: (N, H, W)"

        expand = AsDiscrete(to_onehot=True, n_classes=self.n_classes)
        target = expand(target.unsqueeze(dim=1))  # Shape: (N, C, H, W)

        return (input, target)


LOSSES = {
    "CrossEntropy": CrossEntropyWrapper3D,
    "WeightedCrossEntropy": WeightedCrossEntropyWrapper3D,
    "Focal": FocalLossWrapper3D,
    "Dice": DiceLossWrapper3D,
    "GeneralizedDice": GeneralizedDiceLossWrapper3D,
}


class MultipleLossWrapper3D(MultipleLossWrapper):
    def __init__(self, losses, exclude_missing=False):
        super(MultipleLossWrapper3D, self).__init__(losses, exclude_missing)
