"""
The data modules here are required when working with pytorch lightning.
"""

from multiprocessing import cpu_count
from typing import List, Optional, Union

import pytorch_lightning as pl
from capstone.data.datasets import get_miccai_2d
from capstone.transforms import predefined
from torch.utils.data import DataLoader

DEGREE = {
    1: predefined.windowed_degree_1,
    2: predefined.windowed_degree_2,
    3: predefined.windowed_degree_3,
}


class MiccaiDataModule2D(pl.LightningDataModule):
    def __init__(
        self, batch_size, structure=None, transform_degree: int = None, **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.structure = structure
        assert transform_degree in DEGREE.keys(), "Invalid transform degree passed"
        self.transform = DEGREE[transform_degree]

    def setup(self, stage: Optional[str]):
        if stage == "fit" or stage is None:
            self.train_dataset = get_miccai_2d(
                split="train",
                structure=self.structure,
                transform=self.transform["train"],
            )
            self.val_dataset = get_miccai_2d(
                split="valid",
                structure=self.structure,
                transform=self.transform["test"],
            )

        if stage == "test" or stage is None:
            self.test_dataset = get_miccai_2d(
                split="test",
                structure=self.structure,
                transform=self.transform["test"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
        )
