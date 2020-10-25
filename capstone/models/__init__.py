from capstone.models.losses import BCELossWrapper, DiceLossWrapper
from monai.networks.nets import UNet

LOSSES = {"BCELoss": BCELossWrapper(), "DiceLoss": DiceLossWrapper()}
