from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch.optim as optim
from capstone.data.data_module import MiccaiDataModule2D
from capstone.models import UNet, losses
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.utils import miccai
from pytorch_lightning import Trainer, seed_everything

SEED = 12342


class BaseUNet2D(pl.LightningModule):
    def __init__(
        self,
        filters: Tuple = (16, 32, 64, 128, 256),
        use_res_units: bool = False,
        lr: float = 1e-3,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.unet = UNet(
            dimensions=2,  # 2D images
            in_channels=3,  # if transform_degree in [1, 2, 3]
            out_channels=(
                1 if self._single_structure else len(miccai.STRUCTURES)
            ),  # Predict either 1 or all segmentation masks
            channels=self.hparams.filters,
            strides=(2, 2, 2, 2),  # Default for UNet
            num_res_units=(2 if self.hparams.use_res_units else 0),  # Default in MONAI,
        )
        self.loss_fx = losses.binary_cross_entropy_with_logits  # Default, for now

    @property
    def _single_structure(self):
        return self.hparams.structure is not None

    def forward(self, x):
        x = self.unet(x)
        return x

    def training_step(self, batch, batch_idx):
        images, masks, mask_indicator = batch
        masks = masks.type_as(images)
        mask_indicator = mask_indicator.type_as(images)
        prediction = self.forward(images)
        loss = self.loss_fx(masks, prediction, mask_indicator)
        self.log("loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, mask_indicator = batch
        masks = masks.type_as(images)
        mask_indicator = mask_indicator.type_as(images)
        prediction = self.forward(images)
        loss = self.loss_fx(masks, prediction, mask_indicator)
        self.log("val_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--filters",
            type=tuple,
            default=(16, 32, 64, 128, 256),
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

    parser = BaseUNet2D.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if args.default_root_dir is None:
        args.default_root_dir = DEFAULT_DATA_STORAGE

    main(args)
