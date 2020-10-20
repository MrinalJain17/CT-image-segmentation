import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from capstone.transforms.transforms_2d import WindowedChannels

_stacked_window_stats = {"mean": (0.107, 0.135, 0.085), "std": (0.271, 0.267, 0.152)}
# _no_window_stats = {"mean": (0.077), "std": (0.133)}

_minimal_windowed_test_transform = A.Compose(
    [
        WindowedChannels(),
        A.Resize(256, 256),
        A.Normalize(
            mean=_stacked_window_stats["mean"],
            std=_stacked_window_stats["std"],
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ]
)


def minimal_windowed_transforms(split="train"):
    assert split in ["train", "valid", "test"], "Invalid data split passed"
    if split == "train":
        return A.Compose(
            [
                WindowedChannels(),
                A.RandomCrop(256, 256),
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.Normalize(
                    mean=_stacked_window_stats["mean"],
                    std=_stacked_window_stats["std"],
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return _minimal_windowed_test_transform
