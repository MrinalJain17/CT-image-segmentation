from typing import List, Tuple

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

WINDOWING_CONFIG = {"brain": (80, 40), "soft_tissue": (350, 20), "bone": (2800, 600)}

class WindowedChannels(ImageOnlyTransform):

    def __init__(
        self,
        windows=["brain", "soft_tissue", "bone"],
        shift: bool = True,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(WindowedChannels, self).__init__(always_apply, p)
        self.windows = windows
        self.shift = shift

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        H, W = image.shape[:-1]
        transformed_image = np.empty((H, W, len(self.windows)))

        for i, window in enumerate(self.windows):
            transformed_image[:, :, i] = apply_window(
                image, *WINDOWING_CONFIG[window], shift=self.shift
            ).squeeze(
                axis=2
            )  # Convert transformed image from (H, W, 1) to (H, W)

        return transformed_image  # Shape: (H, W, C), where C is number of window types

    def get_transform_init_args_names(self) -> List:
        return []

def apply_window(
    image: np.ndarray, window_width: int, window_level: int, shift: bool = True
) -> np.ndarray:
    min_ = window_level - (window_width // 2)
    max_ = window_level + (window_width // 2)

    clipped = np.clip(image, min_, max_)
    if shift:
        clipped = (clipped - min_) / (max_ - min_ + 1e-8)

    return clipped

# class WindowingBase(ImageOnlyTransform):
    # """TODO

    # """

    # def __init__(
        # self,
        # window_width: int,
        # window_level: int,
        # shift: bool = True,
        # always_apply: bool = False,
        # p: float = 1.0,
    # ) -> None:
        # super(WindowingBase, self).__init__(always_apply, p)
        # self.window_width = window_width
        # self.window_level = window_level
        # self.shift = shift

    # def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # return apply_window(image, self.window_width, self.window_level, self.shift)

    # def get_transform_init_args_names(self) -> Tuple:
        # return ("window_width", "window_level")


# class BrainWindowing(WindowingBase):
    # def __init__(
        # self, shift: bool = True, always_apply: bool = False, p: float = 1.0
    # ) -> None:
        # super(BrainWindowing, self).__init__(
            # *WINDOWING_CONFIG["brain"], shift=shift, always_apply=always_apply, p=p
        # )


# class SoftTissueWindowing(WindowingBase):
    # def __init__(
        # self, shift: bool = True, always_apply: bool = False, p: float = 1.0
    # ) -> None:
        # super(SoftTissueWindowing, self).__init__(
            # *WINDOWING_CONFIG["soft_tissue"],
            # shift=shift,
            # always_apply=always_apply,
            # p=p
        # )


# class BoneWindowing(WindowingBase):
    # def __init__(
        # self, shift: bool = True, always_apply: bool = False, p: float = 1.0
    # ) -> None:
        # super(BoneWindowing, self).__init__(
            # *WINDOWING_CONFIG["bone"], shift=shift, always_apply=always_apply, p=p
        # )

