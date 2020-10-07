import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import fixed, interact

from .miccai import STRUCTURES, Patient


def plot_slide(patient: Patient, index=0, region=None):
    volume = patient.image.as_numpy()[0]
    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 10))
    axes[0].imshow(volume[index], cmap="gray")
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
            region_array = patient.structures[region[0]].as_numpy()
        else:
            # More than one region requested --> combine
            region_array = patient.combine_structures(structure_list=region)

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
