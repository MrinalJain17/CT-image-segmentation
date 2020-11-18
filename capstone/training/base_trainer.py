from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from capstone.data.data_module import MiccaiDataModule2D, MiccaiDataModule3D
from capstone.models import DiceMetricWrapper, MultipleLossWrapper, UNet
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.training.callbacks import ExamplesLoggingCallback, ExamplesLoggingCallback3D
from capstone.training.utils import _squash_masks,_squash_masks_3D, _squash_predictions
from capstone.utils import miccai
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

SEED = 12342

class BaseUNet(pl.LightningModule):

    @property
    def _n_classes(self):
        return len(miccai.STRUCTURES) + 1  # Additional background
        
    def forward(self, x):
        if self.hparams.downsample:
            x = self.conv1x1(x)
        x = self.unet(x)
        return x
        
    def training_step(self, batch, batch_idx):
        images, masks, mask_indicator, prediction, loss = self._shared_step(
            batch, is_training=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, mask_indicator, prediction, loss = self._shared_step(
            batch, is_training=False
        )
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        """The parameters specific to the model/data processing."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch_size", type=int, default=64, help="Batch size",
        )
        parser.add_argument(
            "--transform_degree",
            type=int,
            default=2,
            help="The degree of transforms/data augmentation to be applied",
        )
        parser.add_argument(
            "--filters",
            nargs=5,
            type=int,
            default=[64, 128, 256, 512, 1024],
            help="A sqeuence of number of filters for the downsampling path in UNet",
        )
        parser.add_argument(
            "--use_res_units",
            action="store_true",
            default=False,
            help="For using residual units in UNet",
        )
        parser.add_argument(
            "--downsample",
            action="store_true",
            default=False,
            help="For using a 1x1 convolution to downsample the input before UNet",
        )
        parser.add_argument(
            "--lr", type=float, default=1e-3, help="Learning rate",
        )
        parser.add_argument(
            "--loss_fx",
            nargs="+",
            type=str,
            default="Dice",
            help="Loss function",
        )
        parser.add_argument(
            "--exclude_missing",
            action="store_true",
            default=False,
            help="Exclude missing annotations from loss computation as described in AnatomyNet",
        )
        
        parser.add_argument(
            "--dimension",
            type=str,
            default='2D',
            help="2D or 3D",
        )
        return parser

class BaseUNet2D(BaseUNet):
    def __init__(
        self,
        filters: List = [16, 32, 64, 128, 256],
        use_res_units: bool = False,
        downsample: bool = False,
        lr: float = 1e-3,
        loss_fx: list = ["CrossEntropy"],
        exclude_missing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        
        assert isinstance(loss_fx, list), "This module expects a list of loss functions"
        loss_fx.sort()  # To have consistent order of loss functions

        self.save_hyperparameters(
            "batch_size",
            "transform_degree",
            "filters",
            "use_res_units",
            "downsample",
            "lr",
            "loss_fx",
            "exclude_missing",
        )
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1)
        self.unet = self._construct_model()
        self.loss_func = MultipleLossWrapper(
            losses=loss_fx, exclude_missing=exclude_missing
        )
        self.dice_score = DiceMetricWrapper()

    def _construct_model(self):
        in_channels = (
            1 if self.hparams.downsample else 3
        )  # assuming transform_degree in [1, 2, 3, 4]
        strides = [2, 2, 2, 2]  # Default for 5-layer UNet

        return UNet(
            dimensions=2,
            in_channels=in_channels,
            out_channels=self._n_classes,
            channels=self.hparams.filters,
            strides=strides,
            num_res_units=(2 if self.hparams.use_res_units else 0),
        )

    def _shared_step(self, batch, is_training: bool):
        images, masks, mask_indicator = batch                                     #Image: Bx3xHxW (64x3x256x256 usually). Mask: #Bx9xHxW (64x9x256x256 usually)
        masks = _squash_masks(masks, self._n_classes, self.device)                #Masks : BxHxW (64x256x256 usually) has number in [0-9] indicating one of 10 classes
        mask_indicator = mask_indicator.type_as(images)
        prefix = "train" if is_training else "val"

        prediction = self.forward(images)                                         #Prediction: Bx10xHxW (64x10x256x256 usually)
        
        loss_dict = self.loss_func(
            input=prediction, target=masks, mask_indicator=mask_indicator
        )
        total_loss = torch.stack(list(loss_dict.values())).sum()

        for name, loss_value in loss_dict.items():
            self.log(
                f"{name} Loss ({prefix})", loss_value, on_step=False, on_epoch=True,
            )

        self._log_dice_scores(prediction, masks, mask_indicator, prefix)
        return images, masks, mask_indicator, prediction, total_loss

    def _log_dice_scores(self, prediction, masks, mask_indicator, prefix):
        pred = prediction.clone()
        self.eval()
        with torch.no_grad():
            if self.hparams.exclude_missing:
                # No indicator for background
                pred[:, 1:, :, :] = pred[:, 1:, :, :] * mask_indicator[:, :, None, None]
            pred = _squash_predictions(pred)  # Shape: (N, H, W)
            dice_mean, dice_per_class = self.dice_score(pred, masks)
            for structure, score in zip(miccai.STRUCTURES, dice_per_class):
                self.log(
                    f"{structure} Dice ({prefix})", score, on_step=False, on_epoch=True,
                )
            self.log(
                f"Mean Dice Score ({prefix})", dice_mean, on_step=False, on_epoch=True,
            )
        self.train()
       
       
class BaseUNet3D(BaseUNet):
    def __init__(
        self,
        filters: List = [16, 32, 64, 128, 256],
        use_res_units: bool = False,
        downsample: bool = False,
        lr: float = 1e-3,
        loss_fx: list = ["Focal"],
        exclude_missing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        
        assert isinstance(loss_fx, list), "This module expects a list of loss functions"
        loss_fx.sort()  # To have consistent order of loss functions
        self.save_hyperparameters(
            "batch_size",
            "transform_degree",
            "filters",
            "use_res_units",
            "downsample",
            "lr",
            "loss_fx",
            "exclude_missing",
        )
        #self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1) #<--- add later if wanted
        self.unet = self._construct_model()
        self.loss_func = MultipleLossWrapper(
            losses=loss_fx, exclude_missing=exclude_missing
        )
        self.dice_score = DiceMetricWrapper()

    def _construct_model(self):
        #in_channels = (
        #    1 if self.hparams.downsample else 3
        #)  # assuming transform_degree in [1, 2, 3, 4]
        strides = [2, 2, 2, 2]  # Default for 5-layer UNet
        in_channels = 1 #change after adding 3D transformations. Now not using any transformations.

        return UNet(
            dimensions=3,
            in_channels=in_channels,
            out_channels=self._n_classes,
            channels=self.hparams.filters,
            strides=strides,
            num_res_units=2,
        )

    def _shared_step(self, batch, is_training: bool):
        images, masks, mask_indicator = batch                                           #Image : Bx1xHxWxD (4x1x256x256x96 usually) #Masks : Bx9xHxWxD (1x9x256x256x96 usually)
                
        masks = _squash_masks_3D(masks, self._n_classes, self.device)                   #Masks: BxHxWxD (4x256x256x96 usually) has number in [0-9] indicating one of 10 classes
        mask_indicator = mask_indicator.type_as(images)
        prefix = "train" if is_training else "val"
        
        prediction = self.forward(images)                                               #Prediction: Bx10xHxWxD (4x10x256x256x96 usually)
        
        loss_dict = self.loss_func(
            input=prediction, target=masks, mask_indicator=mask_indicator
        )
        total_loss = torch.stack(list(loss_dict.values())).sum()

        for name, loss_value in loss_dict.items():
            self.log(
                f"{name} Loss ({prefix})", loss_value, on_step=False, on_epoch=True,
            )
        self._log_dice_scores(prediction, masks, mask_indicator, prefix)
        return images, masks, mask_indicator, prediction, total_loss

    def _log_dice_scores(self, prediction, masks, mask_indicator, prefix):
        pred = prediction.clone()
        self.eval()
        with torch.no_grad():
            #if self.hparams.exclude_missing:                                                 "<---Do later
                # No indicator for background
             #   pred[:, 1:, :, :] = pred[:, 1:, :, :] * mask_indicator[:, :, None, None]
            pred = _squash_predictions(pred)  # Shape: (N, H, W)
            dice_mean, dice_per_class = self.dice_score(pred, masks)
            for structure, score in zip(miccai.STRUCTURES, dice_per_class):
                self.log(
                    f"{structure} Dice ({prefix})", score, on_step=False, on_epoch=True,
                )
            self.log(
                f"Mean Dice Score ({prefix})", dice_mean, on_step=False, on_epoch=True,
            )
        self.train()
        
class WandbLoggerPatch(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params):
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        params = self._sanitize_params(params)
        self.experiment.config.update(params, allow_val_change=True)

def main(args):
    seed_everything(SEED)
    dict_args = vars(args)
    # Data and Model
    if args.dimension == '2D':
        miccai_dm = MiccaiDataModule2D(**dict_args)
        model = BaseUNet2D(**dict_args)
    elif args.dimension == '3D':
        miccai_dm = MiccaiDataModule3D(**dict_args)
        model = BaseUNet3D(**dict_args)
        
    # Trainer
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model=model, datamodule=miccai_dm)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use Weights & Biases for logging",
    )
    parser = BaseUNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    if isinstance(args.loss_fx, str):
        args.loss_fx = [args.loss_fx]

    if args.default_root_dir is None:
        args.default_root_dir = DEFAULT_DATA_STORAGE
    if args.use_wandb:
        if args.dimension == '3D':
            experiment_name = 'UNet 3D'
            args.callbacks = [ExamplesLoggingCallback3D(seed=SEED)]
        else:
            experiment_name = 'UNet 2D'
            args.callbacks = [ExamplesLoggingCallback(seed=SEED)]
        args.logger = WandbLoggerPatch(
            name=experiment_name,
            save_dir=DEFAULT_DATA_STORAGE,
            project="ct-image-segmentation",
        )
        #args.callbacks = [ExamplesLoggingCallback(seed=SEED)]

    main(args)
