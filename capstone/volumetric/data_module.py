from typing import List, Optional, Union

from capstone.volumetric import predefined
from capstone.volumetric.datasets import get_miccai_3d
import pytorch_lightning as pl
from torch.utils.data import DataLoader

DEGREE = {0: predefined.windowed_degree_0}


class MiccaiDataModule3D(pl.LightningDataModule):
    def __init__(self, batch_size, transform_degree: int = None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        assert transform_degree in DEGREE.keys(), "Invalid transform degree passed"
        self.transform = DEGREE[transform_degree]

    def setup(self, stage: Optional[str]):
        if stage == "fit" or stage is None:
            self.train_dataset = get_miccai_3d(
                split="train", transform=self.transform["train"],
            )
            self.val_dataset = get_miccai_3d(
                split="valid", transform=self.transform["test"],
            )

        if stage == "test" or stage is None:
            self.test_dataset = get_miccai_3d(
                split="test", transform=self.transform["test"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
