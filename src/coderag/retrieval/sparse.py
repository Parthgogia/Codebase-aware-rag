"""BM25 retrieval over code chunks, with a tokenizer built for identifiers.

    uv run python -m coderag.retrieval.sparse evaluate-bm25

The tokenizer is the part that matters. Prose tokenizers destroy code: they turn
`_resolve_lookup` into one opaque term that matches nothing a human would type,
or into `resolve`/`lookup` that no longer matches the identifier itself. This
one keeps both.
"""

import keyword
import re
from typing import Any

import bm25s
import pyarrow.parquet as pq
import typer

from coderag.chunking.ast_chunker import AST_CHUNKS_PARQUET
from coderag.chunking.naive import NAIVE_CHUNKS_PARQUET
from coderag.config import settings
from coderag.eval.harness import evaluate, load_queries
from coderag.eval.metrics import RetrievedChunk

app = typer.Typer(add_completion=False, no_args_is_help=True)

# why: identifiers, not words. Splitting on non-alphanumerics but keeping `_`
# holds `__init__` together so it can be emitted whole before being split.
IDENTIFIER = re.compile(r"[A-Za-z0-9_]+")
# why: four alternatives, in this order.
#   [A-Z]+(?=[A-Z][a-z])  an acronym butted against a word: the HTTPS of
#                         HTTPSConnection. Without the lookahead the greedy run
#                         swallows the C and yields HTTP + SConnection.
#   [A-Z]?[a-z0-9]+       an ordinary word: Connection, resolve, x2y.
#   [A-Z]+                a standalone or trailing acronym: ALLCAPS, the HTTP of
#                         get_HTTP_response. Missing this alternative splits
#                         ALLCAPS into seven single letters.
#   [0-9]+                digits left over once letters are taken: HDF5 -> HDF, 5.
CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|[0-9]+")

# why: Python keywords carry no signal -- `self`, `return` and `if` appear in
# every chunk, so they cost index space and dilute BM25's IDF weighting.
KEYWORDS = {word.lower() for word in keyword.kwlist} | {"self", "cls"}


def split_identifier(identifier: str) -> list[str]:
    """`_resolve_lookup` -> ['resolve', 'lookup']; `HTTPSConnection` -> ['HTTPS', 'Connection']."""
    parts: list[str] = []
    for piece in identifier.split("_"):
        parts.extend(CAMEL_PART.findall(piece))
    return parts


def tokenize_code(text: str) -> list[str]:
    """Lowercased tokens: every identifier, plus its constituent parts.

    why: both, not either. See SUMMARY.md -- an issue that says "resolve the
    lookup" must reach `_resolve_lookup`, and an issue that pastes
    `_resolve_lookup` from a traceback must reach it too. Keeping only the parts
    loses the exact match; keeping only the whole loses the prose match.
    """
    tokens: list[str] = []
    for match in IDENTIFIER.finditer(text):
        whole = match.group(0)
        lowered = whole.lower()
        if lowered not in KEYWORDS:
            tokens.append(lowered)
        for part in split_identifier(whole):
            lowered_part = part.lower()
            # why: skip a part identical to the whole, or `sum` would be indexed
            # twice for `sum` and once for `total_sum`, quietly doubling the
            # term frequency of every single-word identifier.
            if lowered_part != lowered and lowered_part not in KEYWORDS:
                tokens.append(lowered_part)
    return tokens


# --- index -------------------------------------------------------------------


class BM25Retriever:
    """A callable `(query, k) -> list[RetrievedChunk]` over one chunk table."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        corpus = [tokenize_code(chunk["text"]) for chunk in chunks]
        self.index = bm25s.BM25()
        self.index.index(corpus, show_progress=False)

    def __call__(self, query: str, k: int) -> list[RetrievedChunk]:
        tokens = tokenize_code(query)
        if not tokens:
            return []
        # why: bm25s returns document indices, not documents, so the chunk table
        # order is the identity map and must not be re-sorted after indexing.
        indices, scores = self.index.retrieve(
            [tokens], k=min(k, len(self.chunks)), show_progress=False
        )
        results = []
        for position in range(indices.shape[1]):
            chunk = self.chunks[int(indices[0, position])]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    file_path=chunk["file_path"],
                    score=float(scores[0, position]),
                    symbols=tuple(chunk["covered_symbols"]),
                )
            )
        return results


@app.callback()
def main() -> None:
    """Sparse retrieval over the chunk index."""
    # why: an explicit callback keeps typer from collapsing the subcommand name.


CHUNK_TABLES = {"naive": NAIVE_CHUNKS_PARQUET, "ast": AST_CHUNKS_PARQUET}


@app.command("evaluate-bm25")
def evaluate_bm25(
    chunks_name: str = typer.Option("naive", "--chunks", help="naive or ast"),
    run_name: str = typer.Option(None, help="Results go to results/<name>.json"),
    text_field: str = typer.Option("query_text", help="query_text or query_text_stripped"),
) -> None:
    """Build the BM25 index over a chunk table and score it."""
    if chunks_name not in CHUNK_TABLES:
        raise typer.BadParameter(f"--chunks must be one of {sorted(CHUNK_TABLES)}")
    run_name = run_name or f"bm25_{chunks_name}"
    chunks = pq.read_table(CHUNK_TABLES[chunks_name]).to_pylist()
    queries = load_queries()
    typer.echo(f"indexing {len(chunks):,} chunks for {len(queries):,} queries...")
    retriever = BM25Retriever(chunks)
    table = evaluate(retriever, queries, run_name, settings.eval_k_values, text_field)
    typer.echo("\n" + table.render())
    typer.echo(f"\nwrote {table.save()}")


if __name__ == "__main__":
    app()
