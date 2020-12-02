import numpy as np
import torch
from capstone.training.utils import _squash_masks, _squash_predictions
from capstone.utils import miccai
from pytorch_lightning.callbacks import Callback
from wandb import Image


class ExamplesLoggingCallback(Callback):
    """Callback to upload sample predictions to W&B."""

    def __init__(self, log_every_n_epochs=25, seed=None) -> None:
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.rng = np.random.default_rng(seed=seed)

    def on_fit_start(self, trainer, pl_module):
        self.sample_indices = self.rng.choice(
            np.arange(pl_module.hparams.batch_size, dtype=int),
            size=min(pl_module.hparams.batch_size, 50),
            replace=False,
        )
        self.num_val_batches = len(trainer.datamodule.val_dataloader())

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (trainer.current_epoch % self.log_every_n_epochs == 0) and (
            (batch_idx + 1) != self.num_val_batches
        ):
            with torch.no_grad():
                sample_images, sample_masks, sample_preds = self._make_predictions(
                    batch, pl_module
                )
                self._log_images(
                    sample_images,
                    sample_masks,
                    sample_preds,
                    pl_module,
                    "val_sample_predictions",
                )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        num_test_batches = len(trainer.datamodule.test_dataloader())
        if (batch_idx + 1) != num_test_batches:
            with torch.no_grad():
                sample_images, sample_masks, sample_preds = self._make_predictions(
                    batch, pl_module
                )
                self._log_images(
                    sample_images,
                    sample_masks,
                    sample_preds,
                    pl_module,
                    "test_sample_predictions",
                )

    def _make_predictions(self, batch, pl_module):
        images, masks, mask_indicator, *others = batch
        images = images.to(pl_module.device)
        masks = masks.to(pl_module.device)
        mask_indicator = mask_indicator.to(pl_module.device).type_as(images)

        masks = _squash_masks(masks, pl_module._n_classes, pl_module.device)

        sample_images = images[self.sample_indices]
        sample_masks = masks[self.sample_indices]
        sample_mask_indicator = mask_indicator[self.sample_indices]
        sample_preds = pl_module.forward(sample_images)
        if pl_module.hparams.exclude_missing:
            # No indicator for background
            sample_preds[:, 1:, :, :] = (
                sample_preds[:, 1:, :, :] * sample_mask_indicator[:, :, None, None]
            )

        return sample_images, sample_masks, sample_preds

    def _log_images(self, images, batch_mask, batch_pred, pl_module, title=None):
        batch_pred = _squash_predictions(batch_pred)  # Shape: (N, H, W)

        class_labels = dict(zip(range(1, pl_module._n_classes), miccai.STRUCTURES))
        class_labels[0] = "Void"

        vis_list = []
        for i, sample in enumerate(images):
            wandb_obj = Image(
                sample.permute(1, 2, 0).detach().cpu().numpy(),
                masks={
                    "predictions": {
                        "mask_data": batch_pred[i].detach().cpu().numpy(),
                        "class_labels": class_labels,
                    },
                    "ground_truth": {
                        "mask_data": batch_mask[i].detach().cpu().numpy(),
                        "class_labels": class_labels,
                    },
                },
            )
            vis_list.append(wandb_obj)

        pl_module.logger.experiment.log(
            {f"{title}": vis_list}, step=pl_module.trainer.global_step
        )
