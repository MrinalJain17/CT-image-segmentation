from capstone.models.temp import GeneralizedDiceLoss
from capstone.utils import miccai
from monai.losses.dice import DiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.transforms import AsDiscrete
import torch
import torch.nn as nn
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

    def __init__(self, **kwargs):
        super(CrossEntropyWrapper, self).__init__()

    @property
    def loss_fx(self):
        return F.cross_entropy

    def _process(self, input, target):
        return (input, target)


class WeightedCrossEntropyWrapper(CrossEntropyWrapper):
    """TODO"""

    def __init__(self, **kwargs):
        super(WeightedCrossEntropyWrapper, self).__init__()
        self.weight = torch.as_tensor(list(WEIGHT.values()))

    def forward(self, input, target):
        input, target = self._process(input, target)
        return self.loss_fx(input, target, weight=self.weight.type_as(input))


class DiceLossWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self, reduction="mean"):
        super(DiceLossWrapper, self).__init__()
        self.reduction = reduction

    @property
    def loss_fx(self):
        return DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            reduction=self.reduction,
        )


class GeneralizedDiceLossWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self, reduction="mean"):
        super(GeneralizedDiceLossWrapper, self).__init__()
        self.reduction = reduction

    @property
    def loss_fx(self):
        return GeneralizedDiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            reduction=self.reduction,
        )


class FocalLossWrapper(BaseLossWrapper):
    """TODO"""

    def __init__(self, reduction="mean"):
        super(FocalLossWrapper, self).__init__()
        self.reduction = reduction
        self.n_classes = len(miccai.STRUCTURES) + 1  # Additional background

    @property
    def loss_fx(self):
        return FocalLoss(reduction=self.reduction)

    def _process(self, input, target):
        assert input.ndim == 4, "Expected input of shape: (N, C, H, W)"
        assert target.ndim == 3, "Expected target of shape: (N, H, W)"

        expand = AsDiscrete(to_onehot=True, n_classes=self.n_classes)
        target = expand(target.unsqueeze(dim=1))  # Shape: (N, C, H, W)

        return (input, target)


class BoundaryLossWrapper(nn.Module):
    """
    Boundary loss between pre-computed distance maps and predictions. Adapted from
    the official repository: https://github.com/LIVIAETS/boundary-loss
    """

    def __init__(self, reduction="mean"):
        super(BoundaryLossWrapper, self).__init__()

        assert reduction in ["none", "mean"]
        self.reduction = reduction

    def forward(self, input, dist_maps):
        input, dist_maps = self._process(input, dist_maps)
        loss = torch.einsum(
            "bchw,bchw->bchw", input[:, 1:, :, :], dist_maps
        )  # Not using background for boundary loss

        if self.reduction == "none":
            return loss.mean(dim=(2, 3))  # Shape: (N, C)

        return loss.mean()  # Scalar

    def _process(self, input, dist_maps):
        assert input.ndim == 4, "Expected input of shape: (N, C, H, W)"
        assert dist_maps.ndim == 4, "Expected distance maps of shape: (N, C, H, W)"

        input = torch.softmax(input, dim=1)
        dist_maps = dist_maps.type_as(input)

        return (input, dist_maps)


LOSSES = {
    "CrossEntropy": CrossEntropyWrapper,
    "WeightedCrossEntropy": WeightedCrossEntropyWrapper,
    "Focal": FocalLossWrapper,
    "Dice": DiceLossWrapper,
    "GeneralizedDice": GeneralizedDiceLossWrapper,
    "Boundary": BoundaryLossWrapper,
}


class MultipleLossWrapper(nn.Module):
    def __init__(self, losses, exclude_missing=False):
        super(MultipleLossWrapper, self).__init__()
        self.exclude_missing = exclude_missing
        for name in losses:
            assert name in LOSSES.keys()

        reduction = "none" if self.exclude_missing else "mean"
        self.losses = nn.ModuleDict(
            {name: LOSSES[name](reduction=reduction) for name in losses}
        )

    def forward(self, input, target, mask_indicator=None, dist_maps=None):
        values = {}
        if mask_indicator is not None:
            mask_indicator = mask_indicator.type_as(input)

        for (name, fx) in self.losses.items():
            if name == "Boundary":
                assert (
                    dist_maps is not None
                ), "Distance maps are required for using boundary loss"
                loss = fx(input, dist_maps)
            else:
                loss = fx(input, target)  # Either scalar or (N, C)

            if self.exclude_missing and (
                name not in ["CrossEntropy", "WeightedCrossEntropy"]
            ):
                loss = apply_missing_mask(name, loss, mask_indicator)

            values[name] = loss

        return values


def apply_missing_mask(name, loss, mask_indicator):
    if name == "Focal":  # Creating mask as described in AnatomyNet
        n_classes = len(miccai.STRUCTURES) + 1
        background_mask = (
            mask_indicator.sum(dim=1, keepdim=True) == (n_classes - 1)
        ).float()
        mask_indicator = torch.cat([background_mask, mask_indicator], dim=1)

    # Weighing by number of annotations per class
    weights = 1.0 / mask_indicator.sum(dim=0)
    if torch.any(torch.isinf(weights)):
        weights = torch.ones_like(weights, device=weights.device)
    weights = weights / weights.sum()

    loss = torch.einsum("ij,j,ij->ij", loss, weights, mask_indicator)
    return loss.sum(dim=1).mean()
