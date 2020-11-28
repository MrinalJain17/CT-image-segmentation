from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from capstone.data.data_module import MiccaiDataModule2D
from capstone.models import DiceMetricWrapper, MultipleLossWrapper, UNet
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.training.callbacks import ExamplesLoggingCallback
from capstone.training.utils import _squash_masks, _squash_predictions
from capstone.utils import miccai
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

SEED = 12342


class BaseUNet2D(pl.LightningModule):
    def __init__(
        self,
        filters: List = [64, 128, 256, 512, 1024],
        use_res_units: bool = False,
        downsample: bool = False,
        lr: float = 1e-2,
        loss_fx: list = ["CrossEntropy"],
        exclude_missing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        assert isinstance(filters, list)
        assert (
            len(filters) == 5
        ), "This module requires a standard 5 block UNet specification"

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

    @property
    def _n_classes(self):
        return len(miccai.STRUCTURES) + 1  # Additional background

    def _construct_model(self):
        in_channels = (
            1
            if (self.hparams.downsample or (self.hparams.transform_degree == 0))
            else 3
        )
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
        if self.hparams.downsample:
            x = self.conv1x1(x)
        x = self.unet(x)
        return x

    def training_step(self, batch, batch_idx):
        _, _, _, _, loss = self._shared_step(batch, is_training=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, is_training=False)

    def _shared_step(self, batch, is_training: bool):
        images, masks, mask_indicator, *dist_maps = batch
        masks = _squash_masks(masks, self._n_classes, self.device)
        mask_indicator = mask_indicator.type_as(images)
        dist_maps = None if (len(dist_maps) == 0) else dist_maps[0]

        prediction = self.forward(images)
        loss_dict = self.loss_func(
            input=prediction,
            target=masks,
            mask_indicator=mask_indicator,
            dist_maps=dist_maps,
        )
        total_loss = torch.stack(list(loss_dict.values())).sum()

        prefix = "train" if is_training else "val"
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

    def configure_optimizers(self):
        return optim.Adagrad(self.parameters(), lr=self.hparams.lr)

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
            default=4,
            help=(
                "The degree of transforms/data augmentation to be applied. "
                "Check 'predefined.py' for available transformations. Note that the "
                "degree here does not represent the strength of transformations. "
                "It's just a way to discern between multiple available options."
            ),
        )
        parser.add_argument(
            "--filters",
            nargs=5,
            type=int,
            default=[64, 128, 256, 512, 1024],
            help="A sqeuence of number of filters for the downsampling path in UNet.",
        )
        parser.add_argument(
            "--use_res_units",
            action="store_true",
            default=False,
            help="For using residual units in UNet.",
        )
        parser.add_argument(
            "--downsample",
            action="store_true",
            default=False,
            help="For using a 1x1 convolution to downsample the input before UNet.",
        )
        parser.add_argument(
            "--lr", type=float, default=1e-2, help="Learning rate",
        )
        parser.add_argument(
            "--loss_fx",
            nargs="+",
            type=str,
            default="CrossEntropy",
            help="Loss function",
        )
        parser.add_argument(
            "--exclude_missing",
            action="store_true",
            default=False,
            help="Exclude missing annotations from loss computation (as described in AnatomyNet).",
        )
        return parser


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

    if "Boundary" in args.loss_fx:
        dict_args["enhanced"] = True

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
        help="Use Weights & Biases for logging.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="UNet 2D",
        help="Experiment name for Weights & Biases.",
    )

    parser = BaseUNet2D.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if isinstance(args.loss_fx, str):
        args.loss_fx = [args.loss_fx]

    if args.default_root_dir is None:
        args.default_root_dir = DEFAULT_DATA_STORAGE

    if args.use_wandb:
        args.logger = WandbLoggerPatch(
            name=args.experiment_name,
            save_dir=DEFAULT_DATA_STORAGE,
            project="ct-image-segmentation",
        )
        args.callbacks = [ExamplesLoggingCallback(seed=SEED)]

    main(args)
