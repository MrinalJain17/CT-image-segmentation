from functools import partial

from capstone.models.temp import compute_meandice, do_metric_reduction
from capstone.utils import miccai
from monai.transforms import AsDiscrete


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
