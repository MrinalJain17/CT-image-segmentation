import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def CT_viewer(volume: np.ndarray, figsize=(9, 6)) -> None:
    if (volume.shape[0] == 1) and (volume.ndim == 4):
        volume = volume[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.volume = volume
    ax1.index = 0
    ax1.imshow(volume[ax1.index], cmap="gray")
    ax1.set_title(f"Slide index: {ax1.index}")

    ax2.hist(ax1.volume[ax1.index].flatten(), bins=20)
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.set_xlabel("Hounsfield Units (HU)")

    axcolor = "lightgoldenrodyellow"
    ax_slide = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    slider = Slider(ax_slide, "Slide", 0, volume.shape[0], valinit=0, valstep=1)

    def update(val):
        ax1.index = slider.val
        ax1.images[0].set_array(volume[ax1.index])
        ax1.set_title(f"Slide index: {ax1.index}")

        ax2.clear()
        ax2.hist(ax1.volume[ax1.index].flatten(), bins=20)
        asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
        ax2.set_aspect(asp)
        ax2.set_xlabel("Hounsfield Units (HU)")

        fig.canvas.draw_idle()

    slider.on_changed(update)
