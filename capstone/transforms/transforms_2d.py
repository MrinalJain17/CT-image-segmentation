from typing import List, Tuple

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

WINDOWING_CONFIG = {"brain": (80, 40), "soft_tissue": (350, 20), "bone": (2800, 600)}


class WindowingBase(ImageOnlyTransform):
    """TODO

    """

    def __init__(
        self,
        window_width: int,
        window_level: int,
        shift: bool = True,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super(WindowingBase, self).__init__(always_apply, p)
        self.window_width = window_width
        self.window_level = window_level
        self.shift = shift

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        return apply_window(image, self.window_width, self.window_level, self.shift)

    def get_transform_init_args_names(self) -> Tuple:
        return ("window_width", "window_level")


class WindowedChannels(ImageOnlyTransform):
    """TODO

    """

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


class BrainWindowing(WindowingBase):
    def __init__(
        self, shift: bool = True, always_apply: bool = False, p: float = 1.0
    ) -> None:
        super(BrainWindowing, self).__init__(
            *WINDOWING_CONFIG["brain"], shift=shift, always_apply=always_apply, p=p
        )


class SoftTissueWindowing(WindowingBase):
    def __init__(
        self, shift: bool = True, always_apply: bool = False, p: float = 1.0
    ) -> None:
        super(SoftTissueWindowing, self).__init__(
            *WINDOWING_CONFIG["soft_tissue"],
            shift=shift,
            always_apply=always_apply,
            p=p
        )


class BoneWindowing(WindowingBase):
    def __init__(
        self, shift: bool = True, always_apply: bool = False, p: float = 1.0
    ) -> None:
        super(BoneWindowing, self).__init__(
            *WINDOWING_CONFIG["bone"], shift=shift, always_apply=always_apply, p=p
        )


def apply_window(
    image: np.ndarray, window_width: int, window_level: int, shift: bool = True
) -> np.ndarray:
    min_ = window_level - (window_width // 2)
    max_ = window_level + (window_width // 2)

    clipped = np.clip(image, min_, max_)
    if shift:
        clipped = (clipped - min_) / (max_ - min_ + 1e-8)

    return clipped
    
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
    Paper called U-Net: Convolutional Networks for Biomedical
    Image Segmentation found elastic transformations to be the most crucial transformation in their segmentation task.
    
    The code below is a demonstration I found in https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation.
    We can adapt this code for our case and experiment how it affects performance later.
    
    --------------------------------------------------------------------------------
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
     -------------------------------------------------------------------------------
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
