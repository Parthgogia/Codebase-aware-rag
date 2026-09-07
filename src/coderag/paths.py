"""Absolute, resolved paths derived from `settings`.

No other module should build a path from a string literal; import from here.
"""

from pathlib import Path

from coderag.config import PROJECT_ROOT, settings


def _resolve(path: Path) -> Path:
    """Anchor a possibly-relative configured path to the project root."""
    # why: config.yaml stores relative paths so the repo is portable, but every
    # consumer wants an absolute path. Absolute entries are passed through so a
    # user can point data_dir at a big external disk.
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


ROOT = PROJECT_ROOT

RAW_DIR = _resolve(settings.raw_dir)
INTERIM_DIR = _resolve(settings.interim_dir)
INDEX_DIR = _resolve(settings.index_dir)
RESULTS_DIR = _resolve(settings.results_dir)

# The pinned checkout of the target repository.
CLONE_DIR = _resolve(settings.clone_dir)

# One JSON file per mined GitHub issue, named <number>.json.
ISSUES_DIR = RAW_DIR / "issues"

# Issue-to-PR linking caches, all keyed by number.
TIMELINES_DIR = RAW_DIR / "timelines"
PULLS_DIR = RAW_DIR / "pulls"
PULL_FILES_DIR = RAW_DIR / "pull_files"

CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def ensure_dirs() -> None:
    """Create the data/results directories if they do not exist yet."""
    for directory in (RAW_DIR, INTERIM_DIR, INDEX_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
