from pathlib import Path
import sys

WORKSPACE_DIR = Path(__file__).parent.parent
DATA = WORKSPACE_DIR / "data"
ARTIFACTS = WORKSPACE_DIR / "artifacts"
SOLUTION = WORKSPACE_DIR
RESOURCES = WORKSPACE_DIR / "resources"
BUILD = WORKSPACE_DIR / "build"

SPLIT_FILE = DATA / "splits.json"

if WORKSPACE_DIR.as_posix() not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR.as_posix())

ARTIFACTS.mkdir(exist_ok=True)