"""Turn each linked PR's patch into the list of symbols it changed.

    uv run python -m coderag.mining.ground_truth build-ground-truth

Reads `data/interim/issue_pr_links.parquet` and the cached patches, resolves
each changed line to the innermost function/class that contained it *before*
the fix, and writes `data/interim/ground_truth.parquet`. These are the
relevance labels every retrieval number in this project is scored against.
"""

import ast
import json
import re
import subprocess
from collections import Counter
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from coderag.mining.linking import LINKS_PARQUET
from coderag.paths import CLONE_DIR, INTERIM_DIR, PULL_FILES_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)

GROUND_TRUTH_PARQUET = INTERIM_DIR / "ground_truth.parquet"
PROGRESS_EVERY = 200

# Sentinels. `<module>` is real module-level code; `<file>` means the symbols
# could not be resolved at all and only the file path is trustworthy.
MODULE_SYMBOL = "<module>"
FILE_SYMBOL = "<file>"

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+")


@app.callback()
def main() -> None:
    """Build the changed-symbol ground truth from linked PR patches."""
    # why: an explicit callback keeps typer from collapsing the subcommand name.


# --- diff parsing ------------------------------------------------------------


def pre_image_lines(patch: str) -> list[int]:
    """Pre-image line numbers this patch touches.

    why: the PRE-image (the `-` side), although the step prompt says post-image.
    The index is built at a commit that predates every one of these fixes, so a
    label naming a line that exists only *after* the fix names code the
    retriever cannot possibly return. Deletions are recorded at their own line;
    an insertion is recorded against the line immediately above it, which is the
    nearest code that actually exists in the indexed tree.
    """
    lines: list[int] = []
    old = 0
    for raw in patch.splitlines():
        header = HUNK_HEADER.match(raw)
        if header:
            old = int(header.group(1))
        elif raw.startswith("-"):
            lines.append(old)
            old += 1
        elif raw.startswith("+"):
            lines.append(max(1, old - 1))
        elif not raw.startswith("\\"):
            # Context line (or the empty string git emits for a blank one).
            old += 1
    return lines


# --- symbols at a commit -----------------------------------------------------


def _show(sha: str, path: str) -> str | None:
    """File content at a commit, or None when it is not present there."""
    # why: a local subprocess rather than repo._git, which strips its output.
    # Stripping would drop leading blank lines and shift every line number in
    # the parsed tree against the line numbers the diff refers to.
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=CLONE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout if result.returncode == 0 else None


def symbols(source: str) -> list[tuple[str, int, int]]:
    """(qualified name, start line, end line) for every def, async def and class.

    Names are qualified within the file only (`DataFrame.merge`); the file path
    is a separate column, so repeating the module in the symbol would duplicate
    it. The join key downstream is therefore (file_path, qualified_name).
    """
    tree = ast.parse(source)
    found: list[tuple[str, int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                # why: start at the first decorator, not the `def`. A patch that
                # only touches `@deprecate(...)` is a change to that function,
                # and anchoring at `def` would credit it to the enclosing class.
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                found.append((name, start, child.end_lineno or child.lineno))
                walk(child, f"{name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return found


def innermost(spans: list[tuple[str, int, int]], line: int) -> str:
    """The tightest symbol containing this line, or `<module>`."""
    best: tuple[str, int, int] | None = None
    for span in spans:
        if span[1] <= line <= span[2]:
            # why: smallest span wins. A line inside DataFrame.merge is inside
            # DataFrame too, but labelling both would make a retriever that
            # returns the 12,000-line class look as good as one that finds the
            # method, which is the distinction this project exists to measure.
            if best is None or (span[2] - span[1]) < (best[2] - best[1]):
                best = span
    return best[0] if best else MODULE_SYMBOL


# --- per-file resolution -----------------------------------------------------


def _rows_for_file(
    issue_number: int, entry: dict[str, Any], parent: str, stats: Counter
) -> Iterator[dict[str, Any]]:
    """Ground-truth rows for one changed file of one PR."""
    # why: for a rename, the pre-image path is the one the index knows about.
    path = entry.get("previous_filename") or entry["filename"]
    patch = entry.get("patch")

    if entry.get("status") == "added":
        # No parent version exists; the file is not in the index at all.
        stats["file_level_added"] += 1
        yield {"issue_number": issue_number, "file_path": path,
               "qualified_name": FILE_SYMBOL, "changed_lines": 0}
        return
    if patch is None:
        stats["file_level_no_patch"] += 1
        yield {"issue_number": issue_number, "file_path": path,
               "qualified_name": FILE_SYMBOL, "changed_lines": 0}
        return

    source = _show(parent, path)
    if source is None:
        stats["file_level_missing_parent"] += 1
        yield {"issue_number": issue_number, "file_path": path,
               "qualified_name": FILE_SYMBOL, "changed_lines": 0}
        return
    try:
        spans = symbols(source)
    except (SyntaxError, ValueError, RecursionError):
        stats["file_level_parse_error"] += 1
        yield {"issue_number": issue_number, "file_path": path,
               "qualified_name": FILE_SYMBOL, "changed_lines": 0}
        return

    stats["files_mapped"] += 1
    counts = Counter(innermost(spans, line) for line in pre_image_lines(patch))
    for name, changed in sorted(counts.items()):
        yield {"issue_number": issue_number, "file_path": path,
               "qualified_name": name, "changed_lines": changed}


# --- command -----------------------------------------------------------------


SCHEMA = pa.schema(
    [
        pa.field("issue_number", pa.int64()),
        pa.field("file_path", pa.string()),
        pa.field("qualified_name", pa.string()),
        pa.field("changed_lines", pa.int64()),
    ]
)


@app.command("build-ground-truth")
def build_ground_truth() -> None:
    """Resolve every linked PR's patch into changed symbols."""
    if not LINKS_PARQUET.exists():
        raise typer.BadParameter(f"{LINKS_PARQUET} missing; run link-issues first")
    if _show("HEAD^", "setup.py") is None and _show("HEAD^", "pyproject.toml") is None:
        raise typer.BadParameter(
            f"{CLONE_DIR} has no history: run `git -C {CLONE_DIR} fetch --unshallow origin`"
        )

    links = pq.read_table(LINKS_PARQUET).to_pylist()
    totals: Counter = Counter()
    stats: Counter = Counter()
    for done, link in enumerate(links, start=1):
        path = PULL_FILES_DIR / f"{link['pr_number']}.json"
        if not path.exists():
            stats["pr_patch_missing"] += 1
            continue
        parent = f"{link['merge_commit_sha']}^"
        for entry in json.loads(path.read_text(encoding="utf-8")):
            if not entry["filename"].endswith(".py"):
                stats["skipped_non_python"] += 1
                continue
            for row in _rows_for_file(int(link["issue_number"]), entry, parent, stats):
                # why: summed, not appended. An issue fixed by two PRs that both
                # touch the same symbol is one label, not two.
                key = (row["issue_number"], row["file_path"], row["qualified_name"])
                totals[key] += int(row["changed_lines"])
        if done % PROGRESS_EVERY == 0:
            typer.echo(f"  {done}/{len(links)} links, {len(totals)} labels")

    rows = [
        {"issue_number": i, "file_path": f, "qualified_name": q, "changed_lines": n}
        for (i, f, q), n in sorted(totals.items())
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), GROUND_TRUTH_PARQUET)
    _report(rows, stats)


def _report(rows: list[dict[str, Any]], stats: Counter) -> None:
    issues = {r["issue_number"] for r in rows}
    file_level = sum(1 for r in rows if r["qualified_name"] == FILE_SYMBOL)
    module_level = sum(1 for r in rows if r["qualified_name"] == MODULE_SYMBOL)
    per_issue = Counter(r["issue_number"] for r in rows)
    typer.echo(f"\nWrote {len(rows):,} labels to {GROUND_TRUTH_PARQUET}")
    typer.echo(f"\nissues with ground truth : {len(issues):,}")
    typer.echo(f"  median symbols per issue: {sorted(per_issue.values())[len(per_issue) // 2]}")
    typer.echo(f"  resolved to a symbol    : {len(rows) - file_level - module_level:,}")
    typer.echo(f"  module-level (<module>) : {module_level:,}")
    typer.echo(f"  file-level only (<file>): {file_level:,}")
    typer.echo("\nfallback reasons and skips:")
    for reason, n in stats.most_common():
        typer.echo(f"  {reason:<26}{n:>8,}")


if __name__ == "__main__":
    app()
