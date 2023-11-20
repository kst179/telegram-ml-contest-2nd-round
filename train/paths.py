from pathlib import Path
import sys

WORKSPACE_DIR = Path(__file__).parent.parent
DATA = Path("data")
ARTIFACTS = Path("artifacts")
SOLUTION = Path(".")
RESOURCES = SOLUTION / "resources"
BUILD = SOLUTION / "build"

SPLIT_FILE = DATA / "splits.json"

if WORKSPACE_DIR.as_posix() not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR.as_posix())

ARTIFACTS.mkdir(exist_ok=True)