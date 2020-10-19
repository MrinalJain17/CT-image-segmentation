from pathlib import Path
from typing import Tuple

import numpy as np
from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.utils import miccai
from torch.utils.data import Dataset


class _MiccaiDataset2D(Dataset):
    def __init__(self, path: str) -> None:
        self.path = Path(path).absolute()

        self.instance_paths = []
        for instance in self.path.iterdir():
            self.instance_paths.append(instance.as_posix())
        self.instance_paths.sort()  # To get same order on Windows and Linux (cluster)

    def __len__(self) -> int:
        return len(self.instance_paths)

    def __getitem__(self, index: int):
        raise NotImplementedError()


class MiccaiDataset2D(_MiccaiDataset2D):
    """TODO

    """

    def __init__(self, path: str, structure: str = None, transform=None) -> None:
        super(MiccaiDataset2D, self).__init__(path)
        if structure is not None:
            assert structure in miccai.STRUCTURES, "Invalid structure name"
        self.structure_required = structure
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        instance = np.load(self.instance_paths[index])
        image = np.transpose(instance["image"], (1, 2, 0))
        masks, mask_indicator = instance["masks"], instance["mask_indicator"]

        assert len(mask_indicator) == 9
        assert masks.shape[0] == 9

        masks = list(masks)

        if self.transform is not None:
            transformed = self.transform(image=image, masks=masks)
            image = transformed["image"]
            masks = transformed["masks"]

        if self.structure_required is not None:
            idx = miccai.STRUCTURES.index(self.structure_required)
            masks = masks[idx]
            mask_indicator = mask_indicator[idx]

        return image, masks, mask_indicator


def get_miccai_2d(
    split: str = "train", structure: str = None, transform=None
) -> Dataset:
    assert split in ["train", "valid", "test"], "Invalid data split passed"
    path = DEFAULT_DATA_STORAGE + f"/miccai_2d/{split}"

    return MiccaiDataset2D(path, structure=structure, transform=transform)
