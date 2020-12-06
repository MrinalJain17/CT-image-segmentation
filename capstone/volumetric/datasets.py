from pathlib import Path
from typing import Tuple

from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.utils import miccai
import numpy as np
import torch
from torch.utils.data import Dataset


class MiccaiDataset3D(Dataset):
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
        image = instance["image"]  # Image : 1xDxHxW
        masks, mask_indicator = (
            instance["masks"],
            instance["mask_indicator"],
        )  # Masks : 9xDxHxW

        assert len(mask_indicator) == len(miccai.STRUCTURES)
        assert masks.shape[0] == len(miccai.STRUCTURES)

        masks = list(masks)
        if self.transform is not None:
            transformed = self.transform(
                image=image, masks=masks
            )  # TO DO: Add 3 new channels!
            image = transformed["image"]  # Image : 1xHxWxD (1x256x256x96 usually)
            masks = transformed["masks"]

        masks = torch.from_numpy(
            np.stack(masks)
        )  # Masks : 9xHxWxD (9x256x256x96 usually)
        mask_indicator = torch.from_numpy(mask_indicator)

        return image, masks, mask_indicator


def get_miccai_3d(split: str = "train", transform=None) -> Dataset:
    assert split in ["train", "valid", "test"], "Invalid data split passed"
    path = DEFAULT_DATA_STORAGE + f"/miccai_3d/{split}"

    return MiccaiDataset3D(path, transform=transform)
