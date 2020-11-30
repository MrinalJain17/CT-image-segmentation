# CT-image-segmentation

:warning: Work In Progress :warning:

Segmentation in Head and Neck CT images

## Installation

Execute the following command from the root of the repository to install the project:

```bash
    pip install -e .
```

**Note that this step is required to run the project.**

## Data

### [MICCAI 2015 Head and Neck Auto Segmentation Challenge](http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge)

Executing the following code will download, extract, and split the dataset.

```bash
    cd capstone/data
    python download.py miccai
```

*Note that if you're on the NYU cluster (Prince), then the data will be stored at the user's `$BEEGFS` directory on the cluster.*
*Refer to the file `paths.py` in the `capstone` directory for more info*

## Requirements

### Base Requirements

1. Python (3.7)
2. [pynrrd](https://github.com/mhe/pynrrd) (0.4) - For loading MICCAI data in `.nrrd` format
3. Tqdm - For displaying progress bars
4. PyTorch (1.7)
5. Torchvision (0.8)
6. [Albumentations](https://github.com/albumentations-team/albumentations) (0.5) - For data augmentation and transforms
7. [MONAI](https://github.com/Project-MONAI/MONAI) (0.3) - For domain specific models, losses, metrics, etc
8. [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (1.0)

### Additional Requirements

1. [Weights and Biases](https://github.com/wandb/client) - For keeping track of experiments
