from argparse import ArgumentParser
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
from capstone.data.data_module import MiccaiDataModule2D
from capstone.models import LOSSES, UNet, losses
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.utils import miccai
from monai.metrics.meandice import DiceMetric
from pytorch_lightning import Trainer, seed_everything
from torchvision.utils import make_grid

SEED = 12342


class BaseUNet2D(pl.LightningModule):
    def __init__(
        self,
        filters: Tuple = (16, 32, 64, 128, 256),
        use_res_units: bool = False,
        lr: float = 1e-3,
        loss_fx: str = "BCELoss",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.unet = self._construct_model()
        self.loss_fx = LOSSES[self.hparams.loss_fx]
        self.dice_score = DiceMetric(sigmoid=True, reduction="none")

    @property
    def _single_structure(self):
        return self.hparams.structure is not None

    def _construct_model(self):
        return UNet(
            dimensions=2,  # 2D images
            in_channels=3,  # assuming transform_degree in [1, 2, 3]
            out_channels=1 if self._single_structure else len(miccai.STRUCTURES),
            channels=self.hparams.filters,
            strides=(2, 2, 2, 2),  # Default for UNet
            num_res_units=(2 if self.hparams.use_res_units else 0),  # Default in MONAI,
        )

    def forward(self, x):
        x = self.unet(x)
        return x

    def training_step(self, batch, batch_idx):
        images, masks, mask_indicator, prediction, loss = self._shared_step(
            batch, batch_idx
        )
        self.log(f"{self.hparams.loss_fx}", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, mask_indicator, prediction, loss = self._shared_step(
            batch, batch_idx
        )
        dice_score = losses._batch_masked_mean(
            self.dice_score(prediction, masks), mask_indicator
        )

        self.log(f"val_{self.hparams.loss_fx}", loss, on_epoch=True)
        self.log("val_DiceScore", dice_score, on_epoch=True)

        if (batch_idx + 1) % (len(self.val_dataloader()) // 4) == 0:
            # 4 batches visualized in every epoch
            self._log_images(masks, prediction)

        return loss

    def _shared_step(self, batch, batch_idx):
        images, masks, mask_indicator = batch
        masks = masks.type_as(images)
        mask_indicator = mask_indicator.type_as(images)
        prediction = self.forward(images)
        loss = self.loss_fx(masks, prediction, mask_indicator)

        return images, masks, mask_indicator, prediction, loss

    def _log_images(self, batch_mask, batch_pred, structure="Mandible"):
        idx = 0 if self._single_structure else miccai.STRUCTURES.index(structure)
        nrow = int(np.sqrt(self.hparams.batch_size))
        mask_grid = make_grid(
            batch_mask[:, [idx], :, :].cpu(), nrow=nrow, pad_value=1
        ).permute((1, 2, 0))
        pred_grid = make_grid(
            batch_pred[:, [idx], :, :].cpu(), nrow=nrow, pad_value=1
        ).permute((1, 2, 0))

        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(16, 8))
        ax[0].imshow(mask_grid[:, :, [0]], cmap="gray")
        ax[0].set_title(f"Actual masks ({structure})")
        ax[1].imshow(pred_grid[:, :, [0]], cmap="gray")
        ax[1].set_title(f"Predicted masks ({structure})")

        self.logger.experiment.add_figure(
            "Actual vs. Predicted Masks", fig, self.trainer.global_step
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
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
        parser.add_argument(
            "--loss_fx",
            type=str,
            default="BCELoss",
            choices=list(LOSSES.keys()),
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

    parser = BaseUNet2D.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if args.default_root_dir is None:
        args.default_root_dir = DEFAULT_DATA_STORAGE

    main(args)
