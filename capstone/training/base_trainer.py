from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
from capstone.data.data_module import MiccaiDataModule2D
from capstone.models import LOSSES, UNet, losses
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.utils import miccai
from monai.metrics.meandice import DiceMetric
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

SEED = 12342


class BaseUNet2D(pl.LightningModule):
    def __init__(
        self,
        filters: List = [16, 32, 64, 128, 256],
        use_res_units: bool = False,
        lr: float = 1e-3,
        loss_fx: str = "BCELoss",
        **kwargs,
    ) -> None:
        super().__init__()
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
        self.loss_func = LOSSES[self.hparams.loss_fx]
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
            strides=[2, 2, 2, 2],  # Default for UNet
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

        _n_batches = 3
        if (batch_idx + 1) % (len(self.val_dataloader()) // _n_batches) == 0:
            # `_n_batches` batches visualized in every epoch
            self._log_images(images, masks, prediction)

        return loss

    def _shared_step(self, batch, batch_idx):
        images, masks, mask_indicator = batch
        masks = masks.type_as(images)
        mask_indicator = mask_indicator.type_as(images)
        prediction = self.forward(images)
        loss = self.loss_func(masks, prediction, mask_indicator)

        return images, masks, mask_indicator, prediction, loss

    def _log_images(self, images, batch_mask, batch_pred):
        if not self._single_structure:
            # Converted masks are compatible for visualization. Shape: (N, H, W)
            convert_values = torch.arange(
                1, len(miccai.STRUCTURES), device=self.device
            )[None, :, None, None]
            converted_mask = (batch_mask * convert_values).sum(dim=1)
            converted_pred = (batch_pred * convert_values).sum(dim=1)

            class_labels = dict(
                zip(range(1, len(miccai.STRUCTURES) + 1), miccai.STRUCTURES)
            )
            class_labels[0] = "Void"
        else:
            converted_mask = batch_mask.squeeze(dim=1)
            converted_pred = batch_pred.squeeze(dim=1)

            class_labels = {0: "Void", 1: self.hparams.structure}

        vis_list = []
        for i, sample in enumerate(images):
            wandb_obj = wandb.Image(
                sample.permute(1, 2, 0).cpu().numpy(),
                masks={
                    "predictions": {
                        "mask_data": converted_pred[i].cpu().numpy(),
                        "class_labels": class_labels,
                    },
                    "ground_truth": {
                        "mask_data": converted_mask[i].cpu().numpy(),
                        "class_labels": class_labels,
                    },
                },
            )
            vis_list.append(wandb_obj)

        self.logger.experiment.log({"sample_predictions": vis_list})

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

    args.logger = WandbLogger(
        "Trial-UNet", DEFAULT_DATA_STORAGE, project="ct-image-segmentation"
    )

    main(args)
