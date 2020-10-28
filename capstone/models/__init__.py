from capstone.models.losses import (
    CrossEntropyWrapper,
    DiceLossWrapper,
    DiceMetricWrapper,
    GeneralizedDiceLossWrapper,
)
from monai.networks.nets import UNet

LOSSES = {
    "CrossEntropy": CrossEntropyWrapper,
    "Dice": DiceLossWrapper,
    "GeneralizedDice": GeneralizedDiceLossWrapper,
}
