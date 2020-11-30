from argparse import ArgumentParser

import torch
from capstone.data.data_module import FullMiccaiDataModule2D, MiccaiDataModule2D
from capstone.models import UNet
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.training.base_trainer import BaseUNet2D, WandbLoggerPatch
from capstone.training.callbacks import ExamplesLoggingCallback
from capstone.training.utils import (
    _squash_masks,
    _squash_predictions,
    mixup_tensors,
    weighted_mixup,
)
from capstone.utils import miccai
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

SEED = 12342


class MixupUNet2D(BaseUNet2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _construct_model(self):
        """1 residual unit works better for mixup."""
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
            num_res_units=(1 if self.hparams.use_res_units else 0),
        )

    def validation_step(self, batch, batch_idx):
        """Mixup is used only while training and not during validation/testing."""
        super()._shared_step(batch, prefix="val")

    def _shared_step(self, batch, prefix: str):
        assert prefix == "train", "Mixup can be used only while training"

        images, masks, mask_indicator, *dist_maps = batch
        mixed_images, shuffle_index, lambda_ = weighted_mixup(
            images, masks, alpha=0.2, device=self.device
        )

        masks = _squash_masks(masks, self._n_classes, self.device)
        dist_maps = None if (len(dist_maps) == 0) else dist_maps[0]

        prediction = self.forward(mixed_images)
        loss_dict_a = self.loss_func(
            input=prediction,
            target=masks,
            mask_indicator=mask_indicator,
            dist_maps=dist_maps,
        )
        loss_dict_b = self.loss_func(
            input=prediction,
            target=masks[shuffle_index],
            mask_indicator=mask_indicator[shuffle_index],
            dist_maps=None if dist_maps is None else dist_maps[shuffle_index],
        )

        loss_dict = {}
        for name in loss_dict_a.keys():
            loss_dict[name] = mixup_tensors(
                loss_dict_a[name], loss_dict_b[name], lambda_
            )
        total_loss = torch.stack(list(loss_dict.values())).sum()

        for name, loss_value in loss_dict.items():
            self.log(
                f"{name} Loss ({prefix})", loss_value, on_step=False, on_epoch=True,
            )

        self._log_mixed_dice_scores(
            prediction, masks, mask_indicator, shuffle_index, lambda_, prefix
        )
        return images, masks, mask_indicator, prediction, total_loss

    def _log_mixed_dice_scores(
        self, prediction, masks, mask_indicator, shuffle_index, lambda_, prefix
    ):
        dice_mean_a, dice_per_class_a = self._get_dice_scores(
            prediction, masks, mask_indicator
        )
        dice_mean_b, dice_per_class_b = self._get_dice_scores(
            prediction, masks[shuffle_index], mask_indicator[shuffle_index]
        )

        for structure, score_a, score_b in zip(
            miccai.STRUCTURES, dice_per_class_a, dice_per_class_b
        ):
            score = mixup_tensors(score_a, score_b, lambda_)
            self.log(
                f"{structure} Dice ({prefix})", score, on_step=False, on_epoch=True,
            )

        dice_mean = mixup_tensors(dice_mean_a, dice_mean_b, lambda_)
        self.log(
            f"Mean Dice Score ({prefix})", dice_mean, on_step=False, on_epoch=True,
        )

    def _get_dice_scores(self, prediction, masks, mask_indicator):
        pred = prediction.clone()
        self.eval()
        with torch.no_grad():
            if self.hparams.exclude_missing:
                # No indicator for background
                pred[:, 1:, :, :] = pred[:, 1:, :, :] * mask_indicator[:, :, None, None]
            pred = _squash_predictions(pred)  # Shape: (N, H, W)
            dice_mean, dice_per_class = self.dice_score(pred, masks)
        self.train()

        return dice_mean, dice_per_class


def main(args):
    seed_everything(SEED)
    dict_args = vars(args)

    if "Boundary" in args.loss_fx:
        dict_args["enhanced"] = True

    # Data
    data_module = FullMiccaiDataModule2D if args.use_full_data else MiccaiDataModule2D
    miccai_2d = data_module(**dict_args)

    # Model
    model = MixupUNet2D(**dict_args)

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
        default="UNet 2D (Mixup)",
        help="Experiment name for Weights & Biases.",
    )

    parser = MixupUNet2D.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if isinstance(args.loss_fx, str):
        args.loss_fx = [args.loss_fx]

    if args.default_root_dir is None:
        args.default_root_dir = DEFAULT_DATA_STORAGE

    args.callbacks = [LearningRateMonitor(logging_interval="epoch")]
    if args.use_wandb:
        args.logger = WandbLoggerPatch(
            name=args.experiment_name,
            save_dir=DEFAULT_DATA_STORAGE,
            project="ct-image-segmentation",
        )
        args.callbacks.append(ExamplesLoggingCallback(seed=SEED))

    main(args)
