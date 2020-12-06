from argparse import ArgumentParser
from pathlib import Path

from capstone.paths import DEFAULT_DATA_STORAGE
from capstone.utils import miccai
import numpy as np


def convert_to_2d(
    read_dir: str, save_dir: str, split: str = None, crop: bool = True,
) -> None:
    """TODO

    """
    read_location = Path(read_dir)
    save_location = Path(save_dir)
    if split is not None:
        read_location = read_location / split
        save_location = save_location / split

    save_location.mkdir(parents=True, exist_ok=True)
    read_location = read_location.as_posix()

    patient_collection = miccai.PatientCollection(read_location)
    _ = patient_collection.apply_function(
        _patient_to_2d, save_location=save_location, crop=crop,
    )


def convert_to_3d(
    read_dir: str, save_dir: str, split: str = None, crop: bool = True,
) -> None:
    """TODO

    """
    read_location = Path(read_dir)
    save_location = Path(save_dir)
    if split is not None:
        read_location = read_location / split
        save_location = save_location / split

    save_location.mkdir(parents=True, exist_ok=True)
    read_location = read_location.as_posix()

    patient_collection = miccai.PatientCollection(read_location)
    _ = patient_collection.apply_function(
        _patient_to_3d, save_location=save_location, crop=crop,
    )


def _patient_to_2d(
    patient: miccai.Patient, save_location: Path, crop: bool = True,
) -> None:
    """TODO

    """
    temp_patient = patient
    if crop:
        temp_patient.crop_data()
    patient_id = Path(temp_patient.patient_dir).stem
    vol = temp_patient.image.as_numpy()

    for index in range(temp_patient.num_slides):
        slide = vol[:, index, :, :]  # Shape: (1, H, W)
        region_slides = []
        mask_indicator = np.ones(len(miccai.STRUCTURES))
        all_zeros = np.zeros_like(
            slide[0, :, :], dtype="uint8"
        )  # Dummy mask. Shape: (H, W)

        for i, structure in enumerate(miccai.STRUCTURES):
            region_volume = temp_patient.structures[structure]
            if region_volume is not None:
                region_slide = region_volume.as_numpy()[0, index, :, :]  # Shape: (H, w)
            else:
                region_slide = all_zeros
                mask_indicator[i] = 0
            region_slides.append(region_slide)

        region_slides = np.stack(
            region_slides
        )  # Shape: (9, H, W) -> 1 mask for each structure

        # Ignore the slide if there is no structure (BrainStem, etc) present
        # It's useless since there's nothing to train/validate on
        if region_slides.sum() > 0:
            filename = (save_location / f"{patient_id}_{index}.npz").as_posix()
            np.savez(
                filename,
                image=slide,
                masks=region_slides,
                mask_indicator=mask_indicator,
            )


def _patient_to_3d(
    patient: miccai.Patient, save_location: Path, crop: bool = True,
) -> None:
    """TODO
    """
    temp_patient = patient
    if crop:
        temp_patient.crop_data()
    patient_id = Path(temp_patient.patient_dir).stem
    vol = temp_patient.image.as_numpy()

    region_slides = []
    mask_indicator = np.ones(len(miccai.STRUCTURES))
    all_zeros = np.zeros_like(
        vol[0, :, :, :], dtype="uint8"
    )  # Dummy mask. Shape: (D, H, W)
    for i, structure in enumerate(miccai.STRUCTURES):
        region_volume = temp_patient.structures[structure]
        if region_volume is not None:
            region_slide = region_volume.as_numpy()[0, :, :, :]  # Shape: (D, H, W)
        else:
            region_slide = all_zeros
            mask_indicator[i] = 0
        region_slides.append(region_slide)

    region_slides = np.stack(
        region_slides
    )  # Shape: (9, D, H, W) -> 1 mask for each structure

    # Ignore the slide if there is no structure (BrainStem, etc) present
    # It's useless since there's nothing to train/validate on
    if region_slides.sum() > 0:
        filename = (save_location / f"{patient_id}.npz").as_posix()
        np.savez(
            filename, image=vol, masks=region_slides, mask_indicator=mask_indicator,
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
        default=None,
        help="Root directory where the train, valid and test splits of MICCAI reside",
    )
    convert_2d_parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory where the converted train, valid and test splits will be saved",
    )
    convert_2d_parser.add_argument(
        "--no_crop",
        action="store_true",
        default=False,
        help="Don't apply the cropping mechanism to volumes before converting",
    )

    convert_3d_parser = subparsers.add_parser("convert_3d")
    convert_3d_parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory where the train, valid and test splits of MICCAI reside",
    )
    convert_3d_parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory where the converted train, valid and test splits will be saved",
    )
    convert_3d_parser.add_argument(
        "--no_crop",
        action="store_true",
        default=False,
        help="Don't apply the cropping mechanism to volumes before converting",
    )

    args = parser.parse_args()
    convert_func = None

    if args.command == "convert_2d":
        if args.root_dir is None:
            args.root_dir = (Path(DEFAULT_DATA_STORAGE) / "miccai").as_posix()
        if args.save_dir is None:
            args.save_dir = (Path(DEFAULT_DATA_STORAGE) / "miccai_2d").as_posix()
        convert_func = convert_to_2d

    elif args.command == "convert_3d":
        if args.root_dir is None:
            args.root_dir = (Path(DEFAULT_DATA_STORAGE) / "miccai").as_posix()
        if args.save_dir is None:
            args.save_dir = (Path(DEFAULT_DATA_STORAGE) / "miccai_3d").as_posix()
        convert_func = convert_to_3d

    if convert_func is not None:
        convert_func(args.root_dir, args.save_dir, "train", not args.no_crop)
        convert_func(args.root_dir, args.save_dir, "valid", not args.no_crop)
        convert_func(args.root_dir, args.save_dir, "test", not args.no_crop)
