import numpy as np
import torch
import wandb
from capstone.utils import miccai
from pytorch_lightning.callbacks import Callback


class ExamplesLoggingCallback(Callback):
    """Callback to upload sample predictions to W&B."""

    def __init__(self, num_examples=50, log_every_n_epochs=10, seed=None) -> None:
        super().__init__()
        self.num_examples = num_examples
        self.log_every_n_epochs = log_every_n_epochs
        self.rng = np.random.default_rng(seed=seed)

    def on_fit_start(self, trainer, pl_module):
        train_dataset = trainer.datamodule.train_dataloader().dataset
        val_dataset = trainer.datamodule.val_dataloader().dataset

        train_samples_idx = self.rng.choice(
            np.arange(len(train_dataset), dtype=int),
            size=self.num_examples,
            replace=False,
        )

        val_samples_idx = self.rng.choice(
            np.arange(len(val_dataset), dtype=int),
            size=self.num_examples,
            replace=False,
        )

        self.train_images, self.train_masks = self._get_sample_data(
            train_dataset, train_samples_idx, pl_module.device
        )
        self.val_images, self.val_masks = self._get_sample_data(
            val_dataset, val_samples_idx, pl_module.device
        )

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            with torch.no_grad():
                pl_module.eval()
                train_pred = pl_module.forward(self.train_images)
                self._log_images(
                    self.train_images,
                    self.train_masks,
                    train_pred,
                    pl_module,
                    "train_sample_predictions",
                )
            pl_module.train()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            with torch.no_grad():
                val_pred = pl_module.forward(self.val_images)
                self._log_images(
                    self.val_images,
                    self.val_masks,
                    val_pred,
                    pl_module,
                    "val_sample_predictions",
                )

    def _get_sample_data(self, dataset, indices, device):
        images = []
        masks = []
        for idx in indices:
            image, mask, _ = dataset[idx]
            images.append(image)
            masks.append(mask)
        images = torch.stack(images).to(device)
        masks = torch.stack(masks).to(device)

        return (images, masks)

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

        pl_module.logger.experiment.log({f"{title}": vis_list})
