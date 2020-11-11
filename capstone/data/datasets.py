"""
These are general pytorch datasets, and can be used to construct data loaders
for training/validation.

It's recommended to use the `get_miccai_2d()` method with appropriate parameters,
which will automatically use the default storage location to load the data.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.utils import miccai
from torch.utils.data import Dataset


class MiccaiDataset2D(Dataset):
    """TODO

    """

    def __init__(self, path: str, transform=None) -> None:
        self.path = Path(path).absolute()
        self.transform = transform

        self.instance_paths = []
        for instance in self.path.iterdir():
            self.instance_paths.append(instance.as_posix())
        self.instance_paths.sort()  # To get same order on Windows and Linux (cluster)

    def __len__(self) -> int:
        return len(self.instance_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        instance = np.load(self.instance_paths[index])
        image = np.transpose(instance["image"], (1, 2, 0))
        masks, mask_indicator = instance["masks"], instance["mask_indicator"]

        assert len(mask_indicator) == len(miccai.STRUCTURES)
        assert masks.shape[0] == len(miccai.STRUCTURES)

        masks = list(masks)

        if self.transform is not None:
            transformed = self.transform(image=image, masks=masks)
            image = transformed["image"]
            masks = transformed["masks"]

        masks = torch.from_numpy(np.stack(masks))
        mask_indicator = torch.from_numpy(mask_indicator)

        return image, masks, mask_indicator


def get_miccai_2d(split: str = "train", transform=None) -> Dataset:
    assert split in ["train", "valid", "test"], "Invalid data split passed"
    path = DEFAULT_DATA_STORAGE + f"/miccai_2d/{split}"

    return MiccaiDataset2D(path, transform=transform)
