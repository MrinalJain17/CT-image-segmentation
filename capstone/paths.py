from pathlib import Path

ROOT = (Path(__file__).resolve().parents[1]).absolute()

DEFAULT_REPO_ROOT = ROOT.as_posix()
DEFAULT_DATA_STORAGE = (ROOT / "storage").as_posix()
