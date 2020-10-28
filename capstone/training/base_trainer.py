from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from capstone.data.data_module import MiccaiDataModule2D
from capstone.models import LOSSES, UNet, losses
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.training.callbacks import ExamplesLoggingCallback
from capstone.utils import miccai
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

SEED = 12342


class BaseUNet2D(pl.LightningModule):
    def __init__(
        self,
        filters: List = [16, 32, 64, 128, 256],
        use_res_units: bool = False,
        lr: float = 1e-3,
        loss_fx: list = ["CrossEntropy"],
        **kwargs,
    ) -> None:
        super().__init__()

        assert isinstance(loss_fx, list), "This module expects a list of loss functions"
        loss_fx.sort()  # To have consistent order of loss functions
        for fx in loss_fx:
            assert fx in LOSSES.keys(), f"Invalid loss function passed: {fx}"

        self.save_hyperparameters(
            "structure",
            "batch_size",
            "transform_degree",
            "filters",
            "use_res_units",
            "lr",
            "loss_fx",
        )
        self.unet = self._construct_model()
        self.loss_func = nn.ModuleList([LOSSES[fx]() for fx in self.hparams.loss_fx])
        self.dice_score = losses.DiceMetricWrapper()

    @property
    def _single_structure(self):
        return self.hparams.structure is not None

    @property
    def _n_classes(self):
        return (
            1 if self._single_structure else len(miccai.STRUCTURES)
        ) + 1  # Additional background

    def _construct_model(self):
        in_channels = 3  # assuming transform_degree in [1, 2, 3]
        strides = [2, 2, 2, 2]  # Default for 5-layer UNet

        return UNet(
            dimensions=2,
            in_channels=in_channels,
            out_channels=self._n_classes,
            channels=self.hparams.filters,
            strides=strides,
            num_res_units=(2 if self.hparams.use_res_units else 0),
        )

    def forward(self, x):
        x = self.unet(x)
        return x

    def training_step(self, batch, batch_idx):
        images, masks, mask_indicator, prediction, loss = self._shared_step(
            batch, batch_idx
        )
        self.log(f"{'+'.join(self.hparams.loss_fx)}", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, mask_indicator, prediction, loss = self._shared_step(
            batch, batch_idx
        )
        dice_score = self.dice_score(prediction, masks)

        self.log(f"val_{'+'.join(self.hparams.loss_fx)}", loss, on_epoch=True)
        self.log("val_DiceScore", dice_score, on_epoch=True)

        return loss

    def _shared_step(self, batch, batch_idx):
        images, masks, mask_indicator = batch

        _temp = torch.arange(1, self._n_classes, device=self.device)
        masks = (masks * _temp[None, :, None, None]).max(dim=1).values

        prediction = self.forward(images)
        loss = sum([fx(prediction, masks) for fx in self.loss_func])

        return images, masks, mask_indicator, prediction, loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """The parameters specific to the model/data processing."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--structure",
            type=str,
            default=None,
            help="A specific structure for the segmentation task",
        )
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
            default=[16, 32, 64, 128, 256],
            help="A sqeuence of number of filters for the downsampling path in UNet",
        )
        parser.add_argument(
            "--use_res_units",
            type=bool,
            default=False,
            help="For using residual units in UNet",
        )
        parser.add_argument(
            "--lr", type=float, default=1e-3, help="Learning rate",
        )
        parser.add_argument(
            "--loss_fx",
            nargs="+",
            type=str,
            default="CrossEntropy",
            help="Loss function",
        )
        return parser


def main(args):
    seed_everything(SEED)
    dict_args = vars(args)

    # Data
    miccai_2d = MiccaiDataModule2D(**dict_args)

    # Model
    model = BaseUNet2D(**dict_args)

    # Trainer
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model=model, datamodule=miccai_2d)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--log_no_examples",
        action="store_true",
        default=False,
        help="Don't log sample predictions in Weights & Biases",
    )

    parser = BaseUNet2D.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if isinstance(args.loss_fx, str):
        args.loss_fx = [args.loss_fx]

    if args.default_root_dir is None:
        args.default_root_dir = DEFAULT_DATA_STORAGE

    if args.use_wandb:
        args.logger = WandbLogger(
            name="UNet 2D",
            save_dir=DEFAULT_DATA_STORAGE,
            project="ct-image-segmentation",
        )
        if not args.log_no_examples:
            args.callbacks = [ExamplesLoggingCallback(seed=SEED)]

    main(args)
