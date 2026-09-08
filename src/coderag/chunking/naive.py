"""Fixed-window chunker: the baseline every code-aware idea must beat.

    uv run python -m coderag.chunking.naive build-naive-chunks

Splits every Python file into overlapping windows of a fixed token budget with
no regard for syntax. Windows land mid-function and mid-statement, which is the
point: Step 9 replaces this with an AST chunker and the difference is the
experiment.
"""

from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
import typer

from coderag.config import settings
from coderag.mining.ground_truth import MODULE_SYMBOL, symbols
from coderag.paths import CLONE_DIR, INDEX_DIR, INTERIM_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)

NAIVE_CHUNKS_PARQUET = INDEX_DIR / "chunks_naive.parquet"
FILES_PARQUET = INTERIM_DIR / "files.parquet"

OVERLAP_TOKENS = 64
# why: cl100k_base is a stand-in for "roughly how long is this", not the
# tokenizer any model here uses. The embedding model in Step 11 has its own and
# the two disagree by 10-20% on code. It is consistent across every chunk, which
# is all a budget needs to be.
ENCODING = "cl100k_base"


@app.callback()
def main() -> None:
    """Build fixed-window chunks over the pinned checkout."""
    # why: an explicit callback keeps typer from collapsing the subcommand name.


def windows(line_tokens: list[int], budget: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Yield (start line, end line), 1-based inclusive, filling a token budget.

    why: windows break on line boundaries rather than exact token counts. A
    chunk starting halfway through a line cannot be cited as `file.py:L120-L155`,
    and the enriched headers, answer citations and permalinks downstream all
    need real line numbers.
    """
    total = len(line_tokens)
    start = 0
    while start < total:
        used = 0
        end = start
        # `end == start` forces progress on a single line longer than the budget.
        while end < total and (used + line_tokens[end] <= budget or end == start):
            used += line_tokens[end]
            end += 1
        yield start + 1, end
        if end >= total:
            return
        back = 0
        cursor = end
        while cursor > start and back < overlap:
            cursor -= 1
            back += line_tokens[cursor]
        start = max(cursor, start + 1)


def covered_symbols(spans: list[tuple[str, int, int]], start: int, end: int) -> list[str]:
    """Symbols whose line range overlaps this window.

    why: overlap, not containment. No 512-token window contains all of a
    2,000-line class, so requiring containment would make bugs in large symbols
    unfindable by any chunker and would be measuring window size, not retrieval.
    """
    hit = [name for name, first, last in spans if first <= end and last >= start]
    # why: a window covering no def at all is still module-level code, and the
    # ground truth labels that the same way.
    return sorted(hit) if hit else [MODULE_SYMBOL]


def _python_files() -> list[str]:
    rows = pq.read_table(FILES_PARQUET).to_pylist()
    return sorted(r["path"] for r in rows if r["extension"] == ".py")


def chunk_file(path: str, encoder: Any, budget: int) -> tuple[list[dict[str, Any]], bool]:
    """Every window of one file, and whether the file failed to parse."""
    text = (CLONE_DIR / path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    if not lines:
        return [], False
    try:
        spans = symbols(text)
        parsed = True
    except (SyntaxError, ValueError, RecursionError):
        # why: a file the parser rejects is still indexable text. It loses its
        # symbol labels, not its chunks, and the count is reported.
        spans, parsed = [], False
    counts = [len(ids) for ids in encoder.encode_ordinary_batch(lines)]
    rows = [
        {
            "chunk_id": f"naive:{path}:{start}-{end}",
            "file_path": path,
            "start_line": start,
            "end_line": end,
            "text": "".join(lines[start - 1 : end]),
            "covered_symbols": covered_symbols(spans, start, end),
            "n_tokens": sum(counts[start - 1 : end]),
        }
        for start, end in windows(counts, budget, OVERLAP_TOKENS)
    ]
    return rows, not parsed


SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("start_line", pa.int64()),
        pa.field("end_line", pa.int64()),
        pa.field("text", pa.string()),
        # why: beyond the columns the step lists. Symbol-level scoring has to
        # know what a window contains, and a fixed window has no single name.
        pa.field("covered_symbols", pa.list_(pa.string())),
        pa.field("n_tokens", pa.int64()),
    ]
)


@app.command("build-naive-chunks")
def build_naive_chunks(
    budget: int = typer.Option(None, help="Tokens per window; defaults to chunk_max_tokens."),
) -> None:
    """Chunk every Python file into fixed windows and write the parquet."""
    budget = budget or settings.chunk_max_tokens
    encoder = tiktoken.get_encoding(ENCODING)
    paths = _python_files()
    typer.echo(
        f"chunking {len(paths):,} Python files at {budget} tokens / {OVERLAP_TOKENS} overlap"
    )

    rows: list[dict[str, Any]] = []
    unparsed = 0
    for done, path in enumerate(paths, start=1):
        file_rows, failed = chunk_file(path, encoder, budget)
        rows.extend(file_rows)
        unparsed += failed
        if done % 300 == 0:
            typer.echo(f"  {done}/{len(paths)} files, {len(rows):,} chunks")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), NAIVE_CHUNKS_PARQUET)
    _report(rows, len(paths), unparsed)


def _report(rows: list[dict[str, Any]], n_files: int, unparsed: int) -> None:
    tokens = sorted(r["n_tokens"] for r in rows)
    per_file: dict[str, int] = {}
    for row in rows:
        per_file[row["file_path"]] = per_file.get(row["file_path"], 0) + 1
    biggest = max(per_file.items(), key=lambda kv: kv[1])
    typer.echo(f"\nWrote {len(rows):,} chunks to {NAIVE_CHUNKS_PARQUET}")
    typer.echo(f"\nfiles chunked      : {n_files:,}")
    typer.echo(f"  failed to parse  : {unparsed:,} (chunked anyway, no symbol labels)")
    typer.echo(
        f"chunks per file    : mean {len(rows) / n_files:.1f}, "
        f"max {biggest[1]} ({biggest[0]})"
    )
    typer.echo(
        f"tokens per chunk   : p50 {tokens[len(tokens) // 2]}, "
        f"p95 {tokens[int(len(tokens) * 0.95)]}, max {tokens[-1]}"
    )


if __name__ == "__main__":
    app()
