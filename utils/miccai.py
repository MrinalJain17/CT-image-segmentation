from pathlib import Path

import nrrd
import numpy as np
import torch
from torchvision.utils import make_grid

from .utils import AttrDict


class Volume(object):
    def __init__(self, path: str):
        self._path = path
        self._data = load_nrrd_as_tensor(path)

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_gray(self) -> bool:
        return True if self.data.shape[0] == 1 else False

    def as_numpy(self, reverse_dims: bool = False) -> np.ndarray:
        arr = self.data.numpy()
        if reverse_dims:
            arr = np.transpose(arr, (2, 3, 1, 0))
        return arr

    def as_grid(
        self, nrow: int = 4, pad_value: int = 1, reverse_dims: bool = True, **kwargs
    ) -> np.ndarray:
        grid = make_grid(
            self.data.permute(1, 0, 2, 3), nrow=nrow, pad_value=pad_value, **kwargs
        )  # Shape: (C, nH, nW)

        if self.is_gray:
            grid = grid[[0]]  # All channels are the same according to the docs
        if reverse_dims:
            grid = grid.permute((1, 2, 0))  # Shape: (nH, nW, C)

        return grid.numpy()


class Patient(object):
    def __init__(self, patient_dir: str):
        self._patient_dir = patient_dir
        self.meta_data = self.get_meta_data()

        self._image = Volume(self.meta_data["image"])
        # self._landmarks = None
        self._structures = self.load_structures()

    @property
    def image(self) -> Volume:
        return self._image

    @property
    def structures(self) -> AttrDict:
        return self._structures

    # @property
    # def landmarks(self):
    #     return self._landmarks

    @property
    def patient_dir(self) -> str:
        return self._patient_dir

    def load_structures(self) -> AttrDict:
        temp = AttrDict()
        for (structure, path) in self.meta_data["structures"].items():
            if path is not None:
                temp[structure] = Volume(path)
            else:
                temp[structure] = None

        return temp

    def get_meta_data(self) -> dict:
        meta_data = {
            "image": None,
            "structures": {
                "BrainStem": None,
                "Chiasm": None,
                "Mandible": None,
                "OpticNerve_L": None,
                "OpticNerve_R": None,
                "Parotid_L": None,
                "Parotid_R": None,
                "Submandibular_L": None,
                "Submandibular_R": None,
            },
            "landmarks": None,
        }
        directory = Path(self.patient_dir)

        meta_data["image"] = (directory / "img.nrrd").as_posix()
        meta_data["landmarks"] = (list(directory.glob("*.fcsv"))[0]).as_posix()

        for structure_path in (directory / "structures").iterdir():
            meta_data["structures"][structure_path.stem] = structure_path.as_posix()

        return meta_data


def load_nrrd_as_tensor(path: str) -> torch.Tensor:
    img, _ = nrrd.read(path)
    if img.ndim == 3:  # grayscale, so adding channel=1
        img = img[:, :, :, np.newaxis]  # Shape: (H, W, D, C)
    tensor = torch.from_numpy(np.transpose(img, (3, 2, 0, 1)))  # Shape: (C, D, H, W)

    return tensor
