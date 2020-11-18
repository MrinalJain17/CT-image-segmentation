import functools
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import nrrd
import numpy as np
import pandas as pd
import torch
from capstone.utils.utils import AttrDict
from torchvision.utils import make_grid
from tqdm import tqdm

# Other parts of the code depend on the order of structures in this list
STRUCTURES: List[str] = [
    "BrainStem",
    "Chiasm",
    "Mandible",
    "OpticNerve_L",
    "OpticNerve_R",
    "Parotid_L",
    "Parotid_R",
    "Submandibular_L",
    "Submandibular_R",
]

LANDMARK_COLS: List[str] = [
    "id",
    "x",
    "y",
    "z",
    "ow",
    "ox",
    "oy",
    "oz",
    "vis",
    "sel",
    "lock",
    "label",
    "desc",
    "associatedNodeID",
]


class Volume(object):
    def __init__(self, path: str = None, data: Union[np.ndarray, torch.Tensor] = None):        
        if path is not None:
            self._path = path
            self._data, self._headers = load_nrrd_as_tensor(path)
        else:
            assert data is not None, "Either one of path or data (array) is required"
            self._path = self._headers = None
            self._data = self._check_data(data)
        self._is_data_modified = False

    def __repr__(self):
        return f"Volume(path={self.path})"

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, arr: Union[np.ndarray, torch.Tensor]) -> None:
        arr = self._check_data(arr)
        self._data = arr
        self._is_data_modified = True

    @property
    def path(self) -> Union[str, None]:
        return self._path

    @property
    def is_gray(self) -> bool:
        return True if self.data.shape[0] == 1 else False

    @property
    def spacing(self) -> Union[np.ndarray, None]:
        if self.headers is not None:
            # Reversed the array to align with channel first format
            # That is, spacing values in dimension: (z, ..., ...)
            return self.headers["space directions"].diagonal()[::-1]
        return None

    def _check_data(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Only meant to be used internally."""
        assert len(data.shape) == 4, "Expected data to be of shape: (C, D, H, W)"
        assert data.shape[0] == 1, "Expected data to be in channel first format"
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        return data

    def _crop_data(
        self, min_z: int, max_z: int, min_x: int, max_x: int, min_y: int, max_y: int
    ) -> None:
        """
        Only intended to be used internally. This function performs no checks,
        and updates the data according to the given crop information (and holds
        no reference to the 'old' data).

        All coordinates are expected to be integers.
        """
        self.data = self.data[:, min_z:max_z, min_x:max_x, min_y:max_y]

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
        self.meta_data = self._store_meta_data()

        self._image = Volume(self.meta_data["image"])
        self._structures = self._load_structures()
        if self.meta_data["landmarks"] is not None:
            self._landmarks = pd.read_csv(
                self.meta_data["landmarks"], comment="#", names=LANDMARK_COLS
            )
        else:  # No landmarks for test data
            self._landmarks = None
        self._is_cropped = False

    def __repr__(self):
        return f"Patient(patient_dir={self.patient_dir})"

    @property
    def image(self) -> Volume:
        return self._image

    @property
    def structures(self) -> AttrDict:
        return self._structures

    @property
    def num_slides(self) -> int:
        return self.image.data.shape[1]

    @property
    def landmarks(self) -> Union[pd.DataFrame, None]:
        return self._landmarks

    @property
    def patient_dir(self) -> str:
        return self._patient_dir

    def _store_meta_data(self) -> Dict:
        meta_data = {
            "image": None,
            "structures": {s: None for s in STRUCTURES},
            "landmarks": None,
        }
        directory = Path(self.patient_dir)
        
        meta_data["image"] = (directory / "img.nrrd").as_posix()
        try:
            meta_data["landmarks"] = (list(directory.glob("*.fcsv"))[0]).as_posix()
        except IndexError:  # No landmarks for test data
            meta_data["landmarks"] = None

        for structure_path in (directory / "structures").iterdir():
            meta_data["structures"][structure_path.stem] = structure_path.as_posix()

        return meta_data

    def _load_structures(self) -> AttrDict:
        temp = AttrDict()
        for (structure, path) in self.meta_data["structures"].items():
            if path is not None:
                temp[structure] = Volume(path)
            else:
                temp[structure] = None
        return temp

    def crop_data(
        self,
        boundary_x: Tuple[int, int] = (120, 400),
        boundary_y: Tuple[int, int] = (55, 335),
        boundary_z: Tuple[float, float] = (0.32, 0.99),
    ):
        assert np.all(
            [isinstance(i, tuple) for i in (boundary_x, boundary_y, boundary_z)]
        ), "Cropping boundary is expected to be a tuple for each axis"

        min_x, max_x = boundary_x
        min_y, max_y = boundary_y
        min_z, max_z = boundary_z

        min_z = np.math.ceil(min_z * self.num_slides)
        max_z = np.math.ceil(max_z * self.num_slides)

        assert np.all(
            [isinstance(i, int) for i in (min_z, max_z, min_x, max_x, min_y, max_y)]
        ), (
            "'x' and 'y' coordinates are expected to be integers, and 'z' "
            "should be float between 0 and 1"
        )
        assert min_x < max_x, "Invalid x-axis boundaries"
        assert min_y < max_y, "Invalid y-axis boundaries"
        assert min_z < max_z, "Invalid z-axis boundaries"

        self.image._crop_data(min_z, max_z, min_x, max_x, min_y, max_y)
        for structure in STRUCTURES:
            if self.structures[structure] is not None:
                self.structures[structure]._crop_data(
                    min_z, max_z, min_x, max_x, min_y, max_y
                )

        self._is_cropped = True

    def combine_segmentation_masks(self, structure_list: list) -> np.ndarray:
        """
        This is used as a workaround for overlaying multiple segmentation masks
        (each corresponding to different region) over a slide. The "correct" way
        can be quite complicated, and is not worth the time for now.
        """
        assert len(structure_list) > 1, "A minimum of 2 structures are required"
        structure_arrays = []

        for structure in structure_list:
            assert structure in STRUCTURES, f"Invalid structure argument: {structure}"
            structure_volume = self.structures[structure]
            if structure_volume is not None:
                structure_arrays.append(structure_volume.as_numpy())

        combined = functools.reduce(np.logical_or, structure_arrays).astype(
            "uint8"
        )  # Shape: (C, D, H, W)
        return combined


class PatientCollection(object):
    def __init__(self, path: str):
        self._path = path
        self._patient_paths = {
            directory.name: directory.as_posix()
            for directory in Path(path).glob("0522c*")
        }
        assert (
            len(self._patient_paths) > 0
        ), "No patients found at the specified location: {path}"

    @property
    def patient_paths(self) -> Dict:
        return self._patient_paths

    def apply_function(
        self, func: Callable, disable_progress: bool = False, **kwargs
    ) -> Dict:
        """
        Applies the callable to each patient, and stores the result in a dictionary.
        Any extra keyword arguments will be passed to the callable.

        The callable should be of the following form:

        def func(patient: Patient, **kwargs):
            ...
        """
        iterator = tqdm(self.patient_paths.items(), disable=disable_progress)

        collected_results = {
            name: func(Patient(path), **kwargs) for (name, path) in iterator
        }

        return collected_results


def load_nrrd_as_tensor(path: str) -> torch.Tensor:
    """
    Headers are returned without any changes. Should be kept in mind if used with the
    tensors, making sure that they both align.
    """
    img, headers = nrrd.read(path)
    if img.ndim == 3:  # grayscale, so adding channel=1
        img = img[:, :, :, np.newaxis]  # Shape: (H, W, D, C)
    tensor = torch.from_numpy(np.transpose(img, (3, 2, 0, 1)))  # Shape: (C, D, H, W)
    return (tensor, headers)
