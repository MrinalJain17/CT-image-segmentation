"""
This file sets the path values depending on where you're working.

It stores:

1. "REPOSITORY_ROOT" variable, points the to root of the repository.
2. "DEFAULT_DATA_STORAGE" variable, points to the directory where the data is stored.

    - If you're on the NYU cluster (prince),
    this will be: "/beegfs/<net_id>/CT-image-segmentation/storage"

    - If not on the cluster, it'll be the directory "storage" in the root of the
    repository.

The code in the entire repository will use these paths by default.
"""

import os
from pathlib import Path


def is_cluster(cluster_type: str = "PRINCE") -> bool:
    """
    Will return True if executed on NYU's HPC Cluster Prince (by default).
    """
    env = os.environ.get("CLUSTER")
    return True if env == cluster_type else False


def _repository_root() -> Path:
    return (Path(__file__).resolve().parents[1]).absolute()


def _storage_root() -> Path:
    if not is_cluster():
        path = _repository_root() / "storage"
    else:
        beegfs_path = Path(os.environ.get("BEEGFS")).absolute()
        path = beegfs_path / "CT-image-segmentation" / "storage"

    return path.absolute()


REPOSITORY_ROOT: str = _repository_root().as_posix()
DEFAULT_DATA_STORAGE: str = _storage_root().as_posix()
