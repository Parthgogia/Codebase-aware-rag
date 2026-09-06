"""Acquire the pinned copy of the target repository and inventory its files.

    uv run python -m coderag.repo clone
    uv run python -m coderag.repo inventory

why subprocess-to-git rather than GitPython: GitPython is a wrapper around the
same `git` binary we would call anyway, so it adds a dependency without adding a
capability. The three commands we need (fetch a single SHA, detached checkout,
rev-parse) are one line each as a subprocess call, and when one fails the error
we want to show the user is git's own stderr, not a GitPython exception.
"""

import subprocess
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from coderag.config import settings
from coderag.paths import CLONE_DIR, INTERIM_DIR, ensure_dirs

app = typer.Typer(add_completion=False, no_args_is_help=True)

FILES_PARQUET = INTERIM_DIR / "files.parquet"


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command, raising with git's own stderr if it fails."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def _head_sha(repo_dir: Path) -> str | None:
    """The commit currently checked out, or None if this is not a git repo."""
    try:
        return _git("rev-parse", "HEAD", cwd=repo_dir)
    except (RuntimeError, FileNotFoundError):
        return None


# --- clone -------------------------------------------------------------------


@app.command()
def clone() -> None:
    """Fetch the pinned commit of the target repo into the configured path."""
    ensure_dirs()
    url = f"https://github.com/{settings.repo_owner}/{settings.repo_name}.git"
    sha = settings.pinned_sha

    if _head_sha(CLONE_DIR) == sha:
        typer.echo(f"Already at {sha[:12]} in {CLONE_DIR}. Nothing to do.")
        return

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    if not (CLONE_DIR / ".git").exists():
        _git("init", "--quiet", str(CLONE_DIR))
        _git("remote", "add", "origin", url, cwd=CLONE_DIR)

    # why: `git clone --depth 1` can only shallow-clone a branch tip, and our SHA
    # is 18 months old. init + fetch of one explicit SHA is the only way to get a
    # depth-1 tree at an arbitrary commit; GitHub allows fetching a SHA directly.
    typer.echo(f"Fetching {sha[:12]} from {url} (depth 1)...")
    _git("fetch", "--depth", "1", "origin", sha, cwd=CLONE_DIR)
    _git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=CLONE_DIR)

    head = _head_sha(CLONE_DIR)
    if head != sha:
        raise RuntimeError(f"checked out {head}, expected {sha}")
    typer.echo(f"Checked out {sha} (detached HEAD) in {CLONE_DIR}.")


# --- inventory ---------------------------------------------------------------


def _matches(rel_path: str, markers: list[str]) -> bool:
    """True if the path hits any marker: '/'-entries are prefixes, others segments."""
    segments = rel_path.split("/")
    for marker in markers:
        if "/" in marker:
            if rel_path == marker or rel_path.startswith(marker + "/"):
                return True
        elif marker in segments:
            return True
    return False


def _is_test(rel_path: str) -> bool:
    """pandas keeps tests in `tests/` packages, plus the usual pytest naming."""
    name = rel_path.split("/")[-1]
    return (
        "tests" in rel_path.split("/")[:-1]
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _line_count(path: Path) -> int:
    """Physical line count, or 0 for binary files."""
    # why: bytes, not text. The tree carries images and .pickle fixtures, and a
    # UnicodeDecodeError in the inventory would mean losing the whole row.
    data = path.read_bytes()
    if not data:
        return 0
    # why: git's own binary test — a NUL byte near the start. Without it the LOC
    # total counts newline bytes inside PNGs and .xlsx fixtures, which makes the
    # summary uncomparable to a cloc/tokei run.
    if b"\x00" in data[:8000]:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _walk_files(root: Path) -> tuple[list[dict[str, object]], int]:
    """Return (kept rows, number of files dropped by the exclude list)."""
    rows: list[dict[str, object]] = []
    excluded = 0
    # why: we do not prune excluded directories from the walk, because the step
    # asks how many files the excludes removed and a pruned directory cannot be
    # counted. The depth-1 fetch keeps .git small enough that this is cheap.
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if _matches(rel, settings.inventory_exclude):
            excluded += 1
            continue
        rows.append(
            {
                "path": rel,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "line_count": _line_count(path),
                "is_test": _is_test(rel),
                "is_vendored": _matches(rel, settings.vendored_markers),
            }
        )
    return rows, excluded


SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("extension", pa.string()),
        pa.field("size_bytes", pa.int64()),
        pa.field("line_count", pa.int64()),
        pa.field("is_test", pa.bool_()),
        pa.field("is_vendored", pa.bool_()),
    ]
)


def _print_summary(rows: list[dict[str, object]], excluded: int) -> None:
    """File count and total LOC per extension, largest extensions first."""
    counts: Counter[str] = Counter()
    loc: Counter[str] = Counter()
    for row in rows:
        ext = str(row["extension"]) or "(none)"
        counts[ext] += 1
        loc[ext] += int(row["line_count"])  # type: ignore[arg-type]

    typer.echo(f"\n{'extension':<14}{'files':>10}{'lines':>12}")
    typer.echo("-" * 36)
    for ext, _ in loc.most_common():
        typer.echo(f"{ext:<14}{counts[ext]:>10,}{loc[ext]:>12,}")
    typer.echo("-" * 36)
    typer.echo(f"{'TOTAL':<14}{sum(counts.values()):>10,}{sum(loc.values()):>12,}")
    tests = sum(1 for r in rows if r["is_test"])
    vendored = sum(1 for r in rows if r["is_vendored"])
    typer.echo(f"\nflagged as test files : {tests:,}")
    typer.echo(f"flagged as vendored   : {vendored:,}")
    typer.echo(f"dropped by excludes   : {excluded:,}")


@app.command()
def inventory() -> None:
    """Walk the pinned checkout and write data/interim/files.parquet."""
    if _head_sha(CLONE_DIR) != settings.pinned_sha:
        raise typer.BadParameter(
            f"{CLONE_DIR} is not checked out at {settings.pinned_sha}. Run `clone` first."
        )
    ensure_dirs()
    rows, excluded = _walk_files(CLONE_DIR)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, FILES_PARQUET)
    typer.echo(f"Wrote {table.num_rows:,} rows to {FILES_PARQUET}")
    _print_summary(rows, excluded)


if __name__ == "__main__":
    app()
