from capstone.models.metrics import DiceMetricWrapper
from monai.transforms import AsDiscrete


class DiceMetricWrapper3D(DiceMetricWrapper):
    """TODO"""

    def __init__(self):
        super(DiceMetricWrapper3D, self).__init__()

    def _process(self, input, target):
        # remove for now, consider making maybe 3D loss function
        # assert input.ndim == 3, "Expected input of shape: (N, H, W)"
        # assert target.ndim == 3, "Expected target of shape: (N, H, W)"

        input = input.unsqueeze(dim=1)  # Shape: (N, 1, H, W)
        target = target.unsqueeze(dim=1)  # Shape: (N, 1, H, W)

        expand = AsDiscrete(to_onehot=True, n_classes=self.n_classes)
        return expand(input), expand(target)  # Shape: (N, C, H, W) - binarized
