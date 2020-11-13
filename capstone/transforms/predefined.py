import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from capstone.transforms.transforms_2d import WindowedChannels

_stacked_window_stats = {"mean": (0.107, 0.135, 0.085), "std": (0.271, 0.267, 0.152)}
_minimal_windowed_transform = A.Compose(
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

windowed_degree_1 = {
    "train": _minimal_windowed_transform,
    "test": _minimal_windowed_transform,
}

windowed_degree_2 = {
    "train": A.Compose(
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
    ),
    "test": _minimal_windowed_transform,
}

windowed_degree_3 = {
    "train": A.Compose(
        [
            WindowedChannels(),
            A.RandomCrop(256, 256),
            A.ElasticTransform(),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.Normalize(
                mean=_stacked_window_stats["mean"],
                std=_stacked_window_stats["std"],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    ),
    "test": _minimal_windowed_transform,
}

windowed_degree_4 = {
    "train": A.Compose(
        [
            WindowedChannels(),
            A.RandomCrop(256, 256),
            A.ElasticTransform(),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.Normalize(
                mean=_stacked_window_stats["mean"],
                std=_stacked_window_stats["std"],
                max_pixel_value=1.0,
            ),
            A.ChannelShuffle(),
            ToTensorV2(),
        ]
    ),
    "test": _minimal_windowed_transform,
}
