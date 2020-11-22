from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from capstone.data.data_module import MiccaiDataModule2D
from capstone.models import DiceMetricWrapper, MultipleLossWrapper, UNet, SegResNetVAE 
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.training.callbacks import ExamplesLoggingCallback
from capstone.training.utils import _squash_masks, _squash_predictions
from capstone.utils import miccai
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

SEED = 12342

class SegResNetVAE2D(pl.LightningModule):
    def __init__(
        self,
        downsample: bool = False,
        lr: float = 1e-3,
        loss_fx: list = ["Focal"],
        exclude_missing: bool = True,
        image_dimensions: tuple = (256,256),
        blocks_down : list = [1,2,2,4],
        blocks_up : list = [1,1,1],
        init_filters: int = 8,
        dropout_prob: float = None,
        use_vae_loss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert isinstance(loss_fx, list), "This module expects a list of loss functions"
        loss_fx.sort()  # To have consistent order of loss functions
        self.save_hyperparameters(
            "batch_size",
            "transform_degree",
            "downsample",
            "lr",
            "loss_fx",
            "exclude_missing",
            "blocks_down",
            "blocks_up",
            "init_filters",
            "dropout_prob",
            "use_vae_loss",
             
        )
        self.image_dimensions = image_dimensions
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.init_filters = init_filters
        self.dropout_prob = dropout_prob
        self.spatial_dimensions = 2
        self.use_vae_loss = use_vae_loss
        
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1)
        self.segnet = self._construct_model()
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

        return SegResNetVAE(
            input_image_size = self.image_dimensions,
            spatial_dims=self.spatial_dimensions,
            in_channels=in_channels,
            out_channels=self._n_classes,
            blocks_down = self.blocks_down,
            blocks_up = self.blocks_up,
            init_filters = self.init_filters,
            dropout_prob = self.dropout_prob
            
        )

    def forward(self, x):
        if self.hparams.downsample:
            x = self.conv1x1(x)
        x = self.segnet(x)
        return x

    def training_step(self, batch, batch_idx):
        _, _, _, _, loss = self._shared_step(batch, is_training=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, is_training=False)

    def _shared_step(self, batch, is_training: bool):
        images, masks, mask_indicator = batch
        masks = _squash_masks(masks, self._n_classes, self.device)
        mask_indicator = mask_indicator.type_as(images)
        prefix = "train" if is_training else "val"
        
        prediction, vae_loss = self.forward(images)
            
        loss_dict = self.loss_func(
            input=prediction, target=masks, mask_indicator=mask_indicator
        )
        total_loss = torch.stack(list(loss_dict.values())).sum()
        
        if is_training:
            if self.use_vae_loss:
                total_loss += vae_loss
            
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
            default=4,
            help=(
                "The degree of transforms/data augmentation to be applied. "
                "Check 'predefined.py' for available transformations. Note that the "
                "degree here does not represent the strength of transformations. "
                "It's just a way to discern between multiple available options."
            ),
        )
        parser.add_argument(
            "--downsample",
            action="store_true",
            default=False,
            help="For using a 1x1 convolution to downsample the input before UNet.",
        )
        parser.add_argument(
            "--lr", type=float, default=1e-3, help="Learning rate",
        )
        parser.add_argument(
            "--loss_fx",
            nargs="+",
            type=str,
            default="Focal",
            help="Loss function",
        )
        parser.add_argument(
            "--exclude_missing",
            action="store_true",
            default=False,
            help="Exclude missing annotations from loss computation (as described in AnatomyNet).",
        )
        parser.add_argument(
            "--blocks_down",
            nargs=4,
            default=[1,2,2,4],
            type = int,
            help="Number of down sample blocks in each layer.",
        )
        parser.add_argument(
            "--blocks_up",
            nargs=3,
            type = int,
            default=[1,1,1],
            help="Number of up sample blocks in each layer.",
        )  
        parser.add_argument(
            "--init_filters",
            type = int,
            default=8,
            help="Number of output channels for initial convolution layer.",
        )      
        parser.add_argument(
            "--dropout_prob",
            type = float,
            default=None,
            help="Probability of an element to be zero-ed.",
        )  

        parser.add_argument(
            "--use_vae_loss",
            action="store_true",
            default=False,
            help="Use VAE loss in addition to prediction loss.",
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

    # Data
    miccai_2d = MiccaiDataModule2D(**dict_args)

    # Model
    model = SegResNetVAE2D(**dict_args)

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
        default="Segnet 2D",
        help="Experiment name for Weights & Biases.",
    )

    parser = SegResNetVAE2D.add_model_specific_args(parser)
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
