##############################################################################
#                The following code is completely experimental
#                ---------------------------------------------
#
# It is an attempt to use "GradCam" to visualize which pixels from the input
# are responsible for predicting a specific class. We use Captum
# (https://github.com/pytorch/captum) for generating the GradCam attributions.
#
##############################################################################

from captum.attr import LayerGradCam
from captum.attr import visualization as viz
import torch
from tqdm import tqdm
import wandb

from capstone.data.data_module import DEGREE
from capstone.data.datasets import get_miccai_2d
from capstone.paths import DEFAULT_DATA_STORAGE, TRAINED_MODELS
from capstone.training import BaseUNet2D, MixupUNet2D
from capstone.training.utils import _squash_masks, _squash_predictions
from capstone.utils import miccai

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
structures = ["Background"] + miccai.STRUCTURES


def get_model(mixup=False):
    model_type = MixupUNet2D if mixup else BaseUNet2D
    weights = TRAINED_MODELS["mixup"] if mixup else TRAINED_MODELS["large"]
    return model_type.load_from_checkpoint(weights).to(device).eval()


def log_samples(image, mask, prediction, class_labels, step):
    wandb_obj = wandb.Image(
        image.permute((1, 2, 0)).detach().cpu().numpy(),
        masks={
            "predictions": {
                "mask_data": prediction.detach().squeeze().cpu().numpy(),
                "class_labels": class_labels,
            },
            "ground_truth": {
                "mask_data": mask.detach().squeeze().cpu().numpy(),
                "class_labels": class_labels,
            },
        },
    )
    wandb.log({"Test predictions": wandb_obj}, step=step)


def main(mixup=False):
    prefix = "Mixup" if mixup else "Large"
    run = wandb.init(
        name=f"Interpretability ({prefix})",
        project="ct-interpretability",
        dir=DEFAULT_DATA_STORAGE,
        reinit=True,
    )
    model = get_model(mixup)
    dataset = get_miccai_2d(
        "test",
        transform=DEGREE[model.hparams.transform_degree]["test"],
        enhanced="Boundary" in model.hparams.loss_fx,
    )

    class_labels = dict(zip(range(1, model._n_classes), miccai.STRUCTURES))
    class_labels[0] = "Void"
    step = 0

    for sample in tqdm(dataset):
        preproc_img, masks, _, *others = sample
        normalized_inp = preproc_img.unsqueeze(0).to(device)
        normalized_inp.requires_grad = True
        masks = _squash_masks(masks, 10, masks.device)

        if len(masks.unique()) < 6:
            # Only displaying structures with atleast 5 structures (excluding background)
            continue

        out = model(normalized_inp)
        out_max = _squash_predictions(out).unsqueeze(1)

        log_samples(preproc_img, masks, out_max, class_labels, step)

        def segmentation_wrapper(input):
            return model(input).sum(dim=(2, 3))

        layer = model.unet.model[2][1].conv.unit0.conv
        lgc = LayerGradCam(segmentation_wrapper, layer)

        figures = []
        for structure in miccai.STRUCTURES:
            idx = structures.index(structure)
            gc_attr = lgc.attribute(normalized_inp, target=idx)
            fig, ax = viz.visualize_image_attr(
                gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                sign="all",
                use_pyplot=False,
            )
            ax.set_title(structure)
            figures.append(wandb.Image(fig))

        wandb.log({"GradCam Attributions": figures}, step=step)
        step += 1

    run.finish()


if __name__ == "__main__":
    main()
    main(mixup=True)
