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

Executing the following code at the root of repository will download, extract, and split the dataset.

```bash
    cd capstone/data
    python download.py miccai
```

## Requirements

1. Python (>=3.7)
2. PyTorch (>=1.6)
3. Torchvision (>=0.7)
4. PyTorch-Lightning (>=1.0.2)
5. Tensorboard
6. [pynrrd](https://github.com/mhe/pynrrd) - For loading MICCAI data in `.nrrd` format
7. Tqdm - For displaying progress bars
