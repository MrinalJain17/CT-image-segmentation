import numpy as np
import torch
import wandb
from capstone.utils import miccai
from pytorch_lightning.callbacks import Callback


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
            size=int(pl_module.hparams.batch_size * self.num_examples),
            replace=False,
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
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
        if trainer.current_epoch % self.log_every_n_epochs == 0:
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

        sample_images = images[self.sample_indices]
        sample_masks = masks[self.sample_indices]
        sample_preds = pl_module.forward(sample_images)

        return sample_images, sample_masks, sample_preds

    def _log_images(self, images, batch_mask, batch_pred, pl_module, title=None):
        # Converting raw predictions to binary using 0.5 as the threshold
        batch_pred = torch.sigmoid(batch_pred)
        batch_pred = (batch_pred > 0.5).type_as(batch_mask)

        if not pl_module._single_structure:
            # Making masks compatible for visualization with wandb. Shape: (N, H, W)
            convert_values = torch.arange(
                1, len(miccai.STRUCTURES) + 1, device=pl_module.device
            )[None, :, None, None]
            converted_mask = (batch_mask * convert_values).max(dim=1).values
            converted_pred = (batch_pred * convert_values).max(dim=1).values

            class_labels = dict(
                zip(range(1, len(miccai.STRUCTURES) + 1), miccai.STRUCTURES)
            )
            class_labels[0] = "Void"
        else:
            converted_mask = batch_mask.squeeze(dim=1)
            converted_pred = batch_pred.squeeze(dim=1)

            class_labels = {0: "Void", 1: pl_module.hparams.structure}

        vis_list = []
        for i, sample in enumerate(images):
            wandb_obj = wandb.Image(
                sample.permute(1, 2, 0).detach().cpu().numpy(),
                masks={
                    "predictions": {
                        "mask_data": converted_pred[i].detach().cpu().numpy(),
                        "class_labels": class_labels,
                    },
                    "ground_truth": {
                        "mask_data": converted_mask[i].detach().cpu().numpy(),
                        "class_labels": class_labels,
                    },
                },
            )
            vis_list.append(wandb_obj)

        pl_module.logger.experiment.log({f"{title}": vis_list})
