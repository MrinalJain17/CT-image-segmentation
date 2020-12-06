from typing import List

from albumentations.core.transforms_interface import DualTransform
import numpy as np
import torch
import torch.nn.functional as F


class Resize3D(DualTransform):
    def __init__(self, size=(96, 256, 256), always_apply=True, p=1.0):
        # DxHxW. Matches our data quite closely (only need to interpolate a little) and works with the existing UNet
        super(Resize3D, self).__init__(always_apply=True, p=1.0)
        self.size = size

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        image = torch.from_numpy(image).unsqueeze(0)
        resized = F.interpolate(image, self.size).squeeze(0)
        return resized

    def apply_to_mask(self, image: np.ndarray, **params) -> np.ndarray:
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(image, self.size).squeeze(0).squeeze(0)
        return resized

    def apply_to_bbox(self, bbox, **params):
        pass

    def apply_to_keypoint(self, keypoint, **params):
        pass

    def get_transform_init_args_names(self) -> List:
        return []


class ToTensorV3(DualTransform):
    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV3, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return img.permute(0, 2, 3, 1)

    def apply_to_mask(self, mask, **params):
        return mask.permute(1, 2, 0)

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}
