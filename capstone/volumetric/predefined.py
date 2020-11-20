import albumentations as A
from capstone.volumetric.transforms import Resize3D, ToTensorV3

windowed_degree_0 = {
    "train": A.Compose([Resize3D(), ToTensorV3()]),
    "test": A.Compose([Resize3D(), ToTensorV3()]),
}
