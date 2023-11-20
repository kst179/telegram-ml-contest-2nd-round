from pathlib import Path

DATA = Path("data")
ARTIFACTS = Path("artifacts")
SOLUTION = Path(".")
RESOURCES = SOLUTION / "resources"
BUILD = SOLUTION / "build"

SPLIT_FILE = DATA / "splits.json"

ARTIFACTS.mkdir(exist_ok=True)