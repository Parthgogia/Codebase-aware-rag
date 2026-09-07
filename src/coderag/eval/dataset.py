"""Assemble the final evaluation query set from issues and ground truth.

    uv run python -m coderag.eval.dataset build-queries

Every retrieval number in this project is measured against the parquet this
writes. The filters are a funnel and each stage prints its count, because a
silently-shrinking eval set is the easiest way to fool yourself.
"""

import json
import re
from collections import Counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from coderag.mining.ground_truth import FILE_SYMBOL, GROUND_TRUTH_PARQUET, MODULE_SYMBOL
from coderag.mining.linking import LINKS_PARQUET
from coderag.paths import INTERIM_DIR, ISSUES_DIR, PULL_FILES_DIR
from coderag.repo import pinned_commit_date

app = typer.Typer(add_completion=False, no_args_is_help=True)

QUERIES_PARQUET = INTERIM_DIR / "eval_queries.parquet"

MAX_PYTHON_FILES = 5
MIN_BODY_CHARS = 50
# why: a symbol shorter than this is a word like `at` or `xs`; matching those
# verbatim in prose would label almost every query a symbol mention. Three still
# admits real API names -- `sum`, `agg`, `map`.
MIN_SYMBOL_CHARS = 3
# why: file stems get a higher bar than symbols. `pytables` or `groupby` really
# is the token a retriever would match, but three-letter stems like `ops` or
# `api` appear in prose constantly and mean nothing.
MIN_STEM_CHARS = 4

TRACEBACK = "traceback"
SYMBOL_MENTION = "symbol_mention"
BEHAVIORAL = "behavioral"

# why: two markers, not one. Users paste tracebacks with the header line cut off
# far more often than you would expect, but the `File "...", line N` frame is
# always there.
TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)|^\s*File \"[^\"]+\", line \d+", re.MULTILINE
)
FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
UNCLOSED_FENCE = re.compile(r"```.*\Z", re.DOTALL)
INDENTED_TRACEBACK = re.compile(
    r"Traceback \(most recent call last\).*?(?=\n\s*\n|\Z)", re.DOTALL
)


@app.callback()
def main() -> None:
    """Build the evaluation query set."""
    # why: an explicit callback keeps typer from collapsing the subcommand name.


# --- text --------------------------------------------------------------------


def query_text(issue: dict[str, Any]) -> str:
    """Title and body, as a developer would have typed the report."""
    return f"{issue.get('title') or ''}\n\n{issue.get('body') or ''}".strip()


def strip_code(text: str) -> str:
    """The query with fenced code blocks and tracebacks removed."""
    # why: a separate column, not a replacement. Step 15 ablates raw against
    # stripped to show how much of the score was traceback leakage rather than
    # retrieval, and that comparison needs both versions of the same query.
    without_blocks = FENCED_BLOCK.sub(" ", text)
    # why: an unterminated fence is common in bug reports (people paste a block
    # and forget to close it). Paired removal leaves the whole tail behind, so
    # anything from a surviving fence to the end goes too.
    without_blocks = UNCLOSED_FENCE.sub(" ", without_blocks)
    return re.sub(r"\n{3,}", "\n\n", INDENTED_TRACEBACK.sub(" ", without_blocks)).strip()


def mentioned_symbols(body: str, files: list[str], symbols: list[str]) -> list[str]:
    """Ground-truth names that appear in the issue text.

    why: matched case-insensitively, and file stems count as well as full
    basenames. The step says "verbatim", but the audit this bucketing exists to
    survive showed 58 supposedly-behavioral queries naming their own ground
    truth as `pytables` or `groupby` -- the exact token BM25 produces from
    `pytables.py`. Leaving those in `behavioral` would inflate the one number
    that is supposed to be honest, so the rule errs toward calling a mention.
    """
    hits: list[str] = []
    candidates: set[str] = set()
    for path in files:
        base = path.rsplit("/", 1)[-1]
        candidates.add(base)
        if base.endswith(".py") and len(base) - 3 >= MIN_STEM_CHARS:
            candidates.add(base[:-3])
    for symbol in symbols:
        if symbol in (MODULE_SYMBOL, FILE_SYMBOL):
            continue
        # why: both the tail and the dotted form. An issue says "fillna", not
        # "NDFrame.fillna", but some do paste the qualified name.
        candidates.add(symbol)
        candidates.add(symbol.rsplit(".", 1)[-1])
    for name in candidates:
        if len(name) < MIN_SYMBOL_CHARS:
            continue
        if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", body, re.IGNORECASE):
            hits.append(name)
    return sorted(hits)


def bucket_for(body: str, files: list[str], symbols: list[str]) -> str:
    """traceback > symbol_mention > behavioral, in that precedence."""
    if TRACEBACK_RE.search(body):
        return TRACEBACK
    if mentioned_symbols(body, files, symbols):
        return SYMBOL_MENTION
    return BEHAVIORAL


# --- inputs ------------------------------------------------------------------


def _python_files(pr_numbers: list[int]) -> tuple[list[str], list[str]]:
    """(changed Python files, of which non-test) across an issue's PRs."""
    changed: set[str] = set()
    for number in pr_numbers:
        path = PULL_FILES_DIR / f"{number}.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            name = entry.get("previous_filename") or entry["filename"]
            if name.endswith(".py"):
                changed.add(name)
    non_test = [
        f
        for f in changed
        if "/tests/" not in f and not f.rsplit("/", 1)[-1].startswith("test_")
    ]
    return sorted(changed), sorted(non_test)


def _load_ground_truth() -> dict[int, list[dict[str, Any]]]:
    labels: dict[int, list[dict[str, Any]]] = {}
    for row in pq.read_table(GROUND_TRUTH_PARQUET).to_pylist():
        labels.setdefault(int(row["issue_number"]), []).append(row)
    return labels


def _load_links() -> dict[int, list[int]]:
    links: dict[int, list[int]] = {}
    for row in pq.read_table(LINKS_PARQUET).to_pylist():
        links.setdefault(int(row["issue_number"]), []).append(int(row["pr_number"]))
    return links


# --- command -----------------------------------------------------------------


SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string()),
        pa.field("issue_number", pa.int64()),
        pa.field("query_text", pa.string()),
        pa.field("query_text_stripped", pa.string()),
        pa.field("bucket", pa.string()),
        pa.field("gt_files", pa.list_(pa.string())),
        pa.field("gt_symbols", pa.list_(pa.string())),
    ]
)


@app.command("build-queries")
def build_queries() -> None:
    """Filter, bucket and write the evaluation query set."""
    labels = _load_ground_truth()
    links = _load_links()
    cutoff = pinned_commit_date()
    funnel: Counter = Counter()
    rows: list[dict[str, Any]] = []

    for path in sorted(ISSUES_DIR.glob("*.json"), key=lambda p: int(p.stem)):
        issue = json.loads(path.read_text(encoding="utf-8"))
        number = int(issue["number"])
        funnel["0. cached closed issues"] += 1

        if (issue.get("closed_at") or "") <= cutoff:
            continue
        funnel["1. closed after the index commit"] += 1

        if number not in links:
            continue
        # why: every link in the table is already confirmed merged by Step 4, so
        # this stage is a pass-through. It is printed anyway because the step
        # asks for it and a silent zero here would mean Step 4 had regressed.
        funnel["2. has a linked merged PR"] += 1

        changed, non_test = _python_files(links[number])
        if not changed or len(changed) > MAX_PYTHON_FILES:
            continue
        funnel[f"3. <= {MAX_PYTHON_FILES} Python files changed"] += 1

        if not non_test:
            continue
        funnel["4. at least one non-test file"] += 1

        if len(issue.get("body") or "") < MIN_BODY_CHARS:
            continue
        funnel[f"5. body >= {MIN_BODY_CHARS} chars"] += 1

        rows_for_issue = labels.get(number, [])
        resolvable = [
            r for r in rows_for_issue
            if r["qualified_name"] not in (MODULE_SYMBOL, FILE_SYMBOL)
        ]
        # why: an extra stage beyond the ones the step lists. An issue whose only
        # label is "<module>" says the fix moved some imports; there is no symbol
        # for a retriever to find, so scoring against it measures nothing.
        if not resolvable:
            continue
        funnel["6. has at least one resolvable symbol"] += 1

        text = query_text(issue)
        gt_files = sorted({r["file_path"] for r in rows_for_issue})
        gt_symbols = sorted({r["qualified_name"] for r in resolvable})
        rows.append(
            {
                "query_id": f"q{number}",
                "issue_number": number,
                "query_text": text,
                "query_text_stripped": strip_code(text),
                "bucket": bucket_for(text, gt_files, gt_symbols),
                "gt_files": gt_files,
                "gt_symbols": gt_symbols,
            }
        )

    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), QUERIES_PARQUET)
    _report(rows, funnel)


def _report(rows: list[dict[str, Any]], funnel: Counter) -> None:
    typer.echo(f"\nWrote {len(rows):,} queries to {QUERIES_PARQUET}\n")
    typer.echo("funnel:")
    previous = None
    for stage in sorted(funnel):
        count = funnel[stage]
        lost = "" if previous is None else f"  (-{previous - count:,})"
        typer.echo(f"  {stage:<38}{count:>7,}{lost}")
        previous = count

    buckets = Counter(r["bucket"] for r in rows)
    typer.echo("\nbuckets:")
    for name in (TRACEBACK, SYMBOL_MENTION, BEHAVIORAL):
        n = buckets[name]
        typer.echo(f"  {name:<18}{n:>7,}  ({n / max(1, len(rows)):.1%})")

    symbols = [len(r["gt_symbols"]) for r in rows]
    stripped = sum(1 for r in rows if len(r["query_text_stripped"]) < len(r["query_text"]))
    typer.echo(f"\nmedian ground-truth symbols per query: {sorted(symbols)[len(symbols) // 2]}")
    typer.echo(f"queries whose text shrank when stripped: {stripped:,}")


if __name__ == "__main__":
    app()
