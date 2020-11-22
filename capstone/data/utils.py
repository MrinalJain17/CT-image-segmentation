"""
Code for computing the distance maps for boundary loss. Adapted from the official
repository: https://github.com/LIVIAETS/boundary-loss
"""

import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    """
    Computes the euclidean distance map of the given segmentation mask.
    Expected shape: (C, H, W)
    """
    num_classes = len(mask)
    result = np.zeros_like(mask)

    for c in range(num_classes):
        posmask = mask[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            result[c] = (distance(negmask) * negmask) - (
                (distance(posmask) - 1) * posmask
            )

    return result / 255.0
