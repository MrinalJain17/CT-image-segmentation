import numpy as np
import torch
from capstone.training.utils import _squash_masks
from capstone.utils import miccai
from pytorch_lightning.callbacks import Callback
from wandb import Image


class ExamplesLoggingCallback(Callback):
    """Callback to upload sample predictions to W&B."""

    def __init__(self, num_examples=0.1, log_every_n_epochs=10, seed=None) -> None:
        super().__init__()
        self.num_examples = num_examples
        self.log_every_n_epochs = log_every_n_epochs
        self.rng = np.random.default_rng(seed=seed)

    def on_fit_start(self, trainer, pl_module):
        self.sample_indices = self.rng.choice(
            np.arange(pl_module.hparams.batch_size, dtype=int),
            size=np.math.ceil(pl_module.hparams.batch_size * self.num_examples),
            replace=False,
        )
        self.num_train_batches = len(trainer.datamodule.train_dataloader())
        self.num_val_batches = len(trainer.datamodule.val_dataloader())

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (trainer.current_epoch % self.log_every_n_epochs == 0) and (
            (batch_idx + 1) != self.num_train_batches
        ):
            with torch.no_grad():
                pl_module.eval()
                sample_images, sample_masks, sample_preds = self._make_predictions(
                    batch, pl_module
                )
                self._log_images(
                    sample_images,
                    sample_masks,
                    sample_preds,
                    pl_module,
                    "train_sample_predictions",
                )
            pl_module.train()

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

    def _make_predictions(self, batch, pl_module):
        images, masks, _ = batch
        images = images.to(pl_module.device)
        masks = masks.to(pl_module.device)

        masks = _squash_masks(masks, pl_module._n_classes, pl_module.device)

        sample_images = images[self.sample_indices]
        sample_masks = masks[self.sample_indices]
        sample_preds = pl_module.forward(sample_images)

        return sample_images, sample_masks, sample_preds

    def _log_images(self, images, batch_mask, batch_pred, pl_module, title=None):
        #  This functions works from PyTorch 1.7 onwards. Will fail for previous
        # versions due to changes in `max()` and `argmax()`
        batch_pred = torch.softmax(batch_pred, dim=1).argmax(dim=1)  # Shape: (N, H, W)

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
