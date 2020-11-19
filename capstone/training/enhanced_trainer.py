from argparse import ArgumentParser
from typing import List

import torch
from capstone.data.data_module import MiccaiDataModule2D
from capstone.models.losses import BoundaryLoss
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.training.base_trainer import BaseUNet2D, WandbLoggerPatch
from capstone.training.callbacks import ExamplesLoggingCallback
from capstone.training.utils import _squash_masks
from pytorch_lightning import Trainer, seed_everything

SEED = 12342


class EnhancedUNet2D(BaseUNet2D):
    def __init__(
        self,
        filters: List = [64, 128, 256, 512, 1024],
        use_res_units: bool = False,
        downsample: bool = False,
        lr: float = 1e-3,
        loss_fx: list = ["CrossEntropy"],
        exclude_missing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            filters, use_res_units, downsample, lr, loss_fx, exclude_missing, **kwargs
        )
        self.compute_boundary_loss = BoundaryLoss()
        self.alpha = 1.0

    def on_train_epoch_end(self, *args, **kwargs):
        """
        'self.alpha' is reduced by 0.005 every epoch. That is, the model will use
        just the boundary loss after 200 epochs.
        """
        super().on_train_epoch_end(*args, **kwargs)
        self.alpha = max(0.005, self.alpha - 0.005)

    def _shared_step(self, batch, is_training: bool):
        images, masks, mask_indicator, dist_maps = batch
        masks = _squash_masks(masks, self._n_classes, self.device)
        mask_indicator = mask_indicator.type_as(images)
        prefix = "train" if is_training else "val"

        prediction = self.forward(images)
        loss_dict = self.loss_func(
            input=prediction, target=masks, mask_indicator=mask_indicator
        )
        total_loss = torch.stack(list(loss_dict.values())).sum()

        # Boundary loss as described in the following paper:
        # https://www.sciencedirect.com/science/article/pii/S1361841520302152?via%3Dihub
        # Note that 'self.alpha' is updated at the end of every epoch.
        boundary_loss = self.compute_boundary_loss(prediction, dist_maps)
        total_loss = (self.alpha * total_loss) + ((1 - self.alpha) * boundary_loss)

        for name, loss_value in loss_dict.items():
            self.log(
                f"{name} Loss ({prefix})", loss_value, on_step=False, on_epoch=True,
            )
        self.log(
            f"Boundary Loss ({prefix})", boundary_loss, on_step=False, on_epoch=True
        )

        self._log_dice_scores(prediction, masks, mask_indicator, prefix)
        return images, masks, mask_indicator, prediction, total_loss


def main(args):
    seed_everything(SEED)
    dict_args = vars(args)

    # Data
    miccai_2d = MiccaiDataModule2D(**dict_args, enhanced=True)

    # Model
    model = EnhancedUNet2D(**dict_args)

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
        "--experiment_name",
        type=str,
        default="Enhanced UNet 2D",
        help="Experiment name for Weights & Biases",
    )

    parser = EnhancedUNet2D.add_model_specific_args(parser)
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
