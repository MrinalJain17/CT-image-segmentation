from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from capstone.utils import miccai


def convert_to_2d(
    read_dir: str = "../../storage/miccai",
    save_dir: str = "../../storage/miccai_2d",
    split: str = None,
    crop: bool = True,
) -> None:
    """TODO

    """
    read_location = Path(read_dir)
    save_location = Path(save_dir)
    if split is not None:
        read_location = read_location / split
        save_location = save_location / split

    save_location.mkdir(parents=True, exist_ok=True)
    image_location = save_location / "images"
    image_location.mkdir(parents=True, exist_ok=True)
    mask_location = save_location / "masks"
    mask_location.mkdir(parents=True, exist_ok=True)

    read_location = read_location.as_posix()
    patient_collection = miccai.PatientCollection(read_location)

    _ = patient_collection.apply_function(
        _patient_to_2d,
        image_location=image_location,
        mask_location=mask_location,
        crop=crop,
    )


def _patient_to_2d(
    patient: miccai.Patient,
    image_location: Path,
    mask_location: Path,
    crop: bool = True,
) -> None:
    """TODO

    """
    temp_patient = patient
    if crop:
        temp_patient.crop_data()
    patient_id = Path(temp_patient.patient_dir).stem

    vol = temp_patient.image.as_numpy()
    for index in range(temp_patient.num_slides):
        slide = vol[:, [index], :, :]
        np.save((image_location / f"{patient_id}_{index}").as_posix(), slide)

        patient_masks_location = mask_location / f"{patient_id}_{index}"
        patient_masks_location.mkdir(parents=True, exist_ok=True)
        for structure in miccai.STRUCTURES:
            region_volume = temp_patient.structures[structure]
            if region_volume is not None:
                region_slide = region_volume.as_numpy()[:, [index], :, :]
                np.save(
                    (patient_masks_location / f"{structure}").as_posix(), region_slide
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="Process MICCAI", dest="command")

    convert_2d_parser = subparsers.add_parser(
        "convert_2d",
        help=(
            "Convert and save the 3D MICCAI patient volumes as 2D images (along with "
            "the segmentation masks)"
        ),
    )
    convert_2d_parser.add_argument(
        "--root_dir",
        type=str,
        default="../../storage/miccai",
        help="Root directory where the train, valid and test splits of MICCAI reside",
    )
    convert_2d_parser.add_argument(
        "--save_dir",
        type=str,
        default="../../storage/miccai_2d",
        help="Directory where the converted train, valid and test splits will be saved",
    )
    convert_2d_parser.add_argument(
        "--no_crop",
        action="store_true",
        default=False,
        help="Don't apply the cropping mechanism to volumes before converting",
    )

    args = parser.parse_args()

    if args.command == "convert_2d":
        convert_to_2d(args.root_dir, args.save_dir, "train", not args.no_crop)
        convert_to_2d(args.root_dir, args.save_dir, "valid", not args.no_crop)
        convert_to_2d(args.root_dir, args.save_dir, "test", not args.no_crop)
