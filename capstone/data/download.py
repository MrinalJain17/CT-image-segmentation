"""Utility for downloading and preparing different datasets

To view all available options:
    python download.py --help

Currently available datasets
----------------------------

1. MICCAI 2015 Head and Neck Segmentation Challange
   (http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge)

    Running the following command will place the contents in the folder
    "storage/miccai" in the root of this repository (default behaviour when not
    on NYU cluster):

        python download.py miccai

    However, if you're on the NYU cluster Prince, then the data will be stored
    (in the same structure) in the user's "$BEEGFS" directory, in a folder
    "CT-image-segmentation".

    Or you could specify the directory explicitly (not recommended):

        python download.py miccai --root_dir ~/capstone/data

"""

from argparse import ArgumentParser
from pathlib import Path
import shutil

from capstone.paths import DEFAULT_DATA_STORAGE
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive

SEED = 42


def prepare_miccai(root_dir: str, download: bool = True) -> None:
    """MICCAI 2015 Head and Neck Segmentation Dataset

    Downloads the dataset and performs the split into train, validation, and test
    sets as described in http://www.imagenglab.com/newsite/pddca/
    """
    urls = {
        "part-1": "http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part1.zip",
        "part-2": "http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part2.zip",
        "part-3": "http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part3.zip",
    }

    path = root_dir

    if download:
        for (_, url) in urls.items():
            download_and_extract_archive(
                url=url, download_root=path, remove_finished=True
            )

    path = Path(path)
    patients = [directory for directory in path.glob("0522c*")]
    patients.sort()  # To get same splitting on Windows and Linux (cluster)

    # Dataset consists CT scans of 48 patients
    assert len(patients) == 48, (
        f"The required patient directories of MICCAI dataset not found at the "
        f"given path: {path.absolute()}"
    )

    rng = np.random.default_rng(seed=SEED)

    # Splitting patient directories into train and test sets as
    # described in http://www.imagenglab.com/newsite/pddca/
    test = range(555, 879)
    train = [
        int(directory.name[-4:])
        for directory in patients
        if int(directory.name[-4:]) in range(1, 480)
    ]
    rng.shuffle(train)
    valid = train[:8]
    train = train[8:]

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
        if args.root_dir is None:
            args.root_dir = (Path(DEFAULT_DATA_STORAGE) / "miccai").as_posix()
        prepare_miccai(args.root_dir, not args.no_download)
