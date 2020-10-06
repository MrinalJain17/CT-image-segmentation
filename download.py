"""Utility for downloading and preparing different datasets

To view all available options:
    python download.py --help

Currently available datasets
----------------------------

1. MICCAI 2015 Head and Neck Segmentation Challange
   (http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge)

    Running the following command will place the contents in the folder
    "data/miccai" in the current working directory (default behaviour):

        python download.py miccai

    Or you could specify the directory explicitly:

        python download.py miccai --root_dir ~/capstone/data

"""

import shutil
from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive


def prepare_miccai(root_dir=None, download=True):
    """MICCAI 2015 Head and Neck Segmentation Dataset

    Downloads the dataset and performs the split into train, validation, and test
    sets as described in http://www.imagenglab.com/newsite/pddca/
    """
    urls = {
        "part-1": "http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part1.zip",
        "part-2": "http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part2.zip",
        "part-3": "http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part3.zip",
    }

    path = "./data/miccai" if root_dir is None else root_dir

    if download:
        for (_, url) in urls.items():
            download_and_extract_archive(
                url=url, download_root=path, remove_finished=True
            )

    path = Path(path)
    patients = [directory for directory in path.glob("0522c*")]

    # Dataset consists CT scans of 48 patients
    assert len(patients) == 48, (
        f"The required patient directories of MICCAI dataset not found at the "
        f"given path: {path.absolute()}"
    )

    # Splitting patient directories into train, test and validation sets as
    # described in http://www.imagenglab.com/newsite/pddca/
    train = range(1, 329)
    valid = range(329, 480)
    test = range(555, 879)

    for patient in patients:
        num = int(patient.name[5:])
        split = ""
        if num in train:
            split = "train"
        elif num in valid:
            split = "valid"
        elif num in test:
            split = "test"

        _ = shutil.move(patient, path / split / patient.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="Available datasets", dest="command")

    miccai_parser = subparsers.add_parser(
        "miccai",
        help="Download and prepare the MICCAI 2015 Head and Neck Cancer Dataset",
    )
    miccai_parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory for storing the downloaded/extracted data",
    )
    miccai_parser.add_argument(
        "--no_download",
        action="store_true",
        default=False,
        help="Don't download the dataset",
    )

    args = parser.parse_args()

    if args.command == "miccai":
        prepare_miccai(args.root_dir, not args.no_download)
