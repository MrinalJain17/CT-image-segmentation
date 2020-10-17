import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from capstone.utils.miccai import STRUCTURES, Patient
from ipywidgets import fixed, interact


def plot_slide(patient: Patient, index: int = 0, region=None):
    volume = patient.image.as_numpy()[0]
    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 10))
    axes[0].imshow(volume[index], cmap=plt.cm.bone)
    axes[0].set_title(f"Slide index: {index}")

    axes[1].hist(volume[index].flatten(), bins=20)
    axes[1].set_xlabel("Hounsfield Units (HU)")
    axes[1].set_aspect(1.0 / axes[1].get_data_ratio())

    if region not in [None, ()]:
        if isinstance(region, str):
            assert (
                region in patient.structures.keys()
            ), f"Invalid region argument: {region}"
            region = [region]

        if len(region) == 1:
            region_array = patient.structures[region[0]]
            if region_array is not None:
                region_array = region_array.as_numpy()
            else:
                region_array = np.expand_dims(np.zeros_like(volume), 0)
        else:
            # More than one region requested --> combine
            region_array = patient.combine_segmentation_masks(structure_list=region)

        axes[0].imshow(region_array[0][index], alpha=0.5)

    return axes


def notebook_interact(patient: Patient):
    return interact(
        plot_slide,
        patient=fixed(patient),
        index=widgets.IntSlider(
            value=0,
            min=0,
            max=(patient.num_slides - 1),
            step=1,
            continuous_update=False,
        ),
        region=widgets.SelectMultiple(
            options=STRUCTURES, value=(), description="Region: ", disabled=False,
        ),
    )


def plot_region_distribution(patient: Patient, exclude=None, ax=None):
    voxel_dict = {}
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    for structure in STRUCTURES:
        if structure not in exclude:
            region_volume = patient.structures[structure]
            if region_volume is not None:
                voxel_dict[structure] = patient.image.as_numpy()[
                    region_volume.as_numpy() == 1
                ]

    ax = sns.boxplot(
        data=list(voxel_dict.values()), showmeans=True, showfliers=False, ax=ax
    )
    ax.set_xticklabels(voxel_dict.keys(), rotation=45)
    ax.set_ylabel("Hounsfield Units (HU)")

    return ax


def plot_windowed(patient: Patient, index: int = 0):
    configs = {
        "brain": (80, 40),
        "subdural": (215, 75),
        "stroke": (8, 32),
        "temporal bones": (2800, 600),
        "soft tissues": (375, 40),
    }  # Values taken from https://radiopaedia.org/articles/windowing-ct

    num_cols = 2
    num_rows = len(configs) + 1
    fig, axes = plt.subplots(
        num_rows, num_cols, constrained_layout=True, figsize=(14, 5 * num_rows)
    )

    axes[0, 0].imshow(patient.image.as_numpy()[0][index], cmap=plt.cm.bone)
    axes[0, 0].set_title(f"Original slide: {index}")

    axes[0, 1].hist(patient.image.as_numpy()[0][index].flatten(), bins=20)
    axes[0, 1].set_xlabel("Hounsfield Units (HU)")
    axes[0, 1].set_aspect(1.0 / axes[0, 1].get_data_ratio())

    for i, (window_type, window) in enumerate(configs.items()):
        windowed_volume = _window_volume(patient, *window)

        axes[i + 1, 0].imshow(windowed_volume[0][index], cmap=plt.cm.bone)
        axes[i + 1, 0].set_title(f"{window_type}: width={window[0]}, level={window[1]}")

        axes[i + 1, 1].hist(windowed_volume[0][index].flatten(), bins=20)
        axes[i + 1, 1].set_xlabel("Hounsfield Units (HU)")
        axes[i + 1, 1].set_aspect(1.0 / axes[i + 1, 1].get_data_ratio())

    return axes


def _window_volume(
    patient: Patient, window_width: int, window_level: int
) -> np.ndarray:
    """
    Function to apply a windowing operation. This implemetation is intended to be
    used for visualization only.
    """
    assert isinstance(patient, Patient)

    min_ = window_level - (window_width // 2)
    max_ = window_level + (window_width // 2)

    volume = patient.image.as_numpy()
    volume = np.clip(volume, min_, max_)

    return volume
