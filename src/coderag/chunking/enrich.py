"""Prepend a self-describing header to every AST chunk.

    uv run python -m coderag.chunking.enrich build-enriched-chunks

A raw chunk is a body of code with no idea where it came from. `NDFrame.fillna`
retrieved on its own does not say it belongs to `DataFrame`, does not say the
module is about generic frame operations, and does not repeat its own signature
if the window starts below it. The header says all of that in a few lines, and
those lines are exactly the vocabulary a bug report is written in.
"""

import ast
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
import typer

from coderag.chunking.ast_chunker import AST_CHUNKS_PARQUET, SCHEMA as AST_SCHEMA
from coderag.chunking.naive import ENCODING
from coderag.config import settings
from coderag.paths import CLONE_DIR, INDEX_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)

ENRICHED_CHUNKS_PARQUET = INDEX_DIR / "chunks_ast_enriched.parquet"
SEPARATOR = "---"
# why: dropped in this order when the header exceeds its budget. The file path
# and signature are what identify the chunk; the import list is context that a
# query will rarely match on, so it goes first.
DROP_ORDER = ("Imports", "Module", "Class")


@app.callback()
def main() -> None:
    """Build enriched chunk text."""
    # why: an explicit callback keeps typer from collapsing the subcommand name.


def first_line(docstring: str | None) -> str:
    """A docstring truncated to its first non-empty line."""
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def render_imports(tree: ast.Module) -> list[str]:
    """Deduplicated import targets, in source order."""
    seen: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            rendered = [
                f"{a.name} as {a.asname}" if a.asname else a.name for a in node.names
            ]
        elif isinstance(node, ast.ImportFrom):
            # why: the module, not the imported names. `pandas.core.dtypes.common`
            # is the useful term; listing forty imported helpers would blow the
            # budget and add vocabulary no bug report uses.
            rendered = ["." * node.level + (node.module or "")]
        else:
            continue
        for item in rendered:
            if item and item not in seen:
                seen.append(item)
    return seen


def definitions(tree: ast.Module, lines: list[str]) -> dict[str, dict[str, str]]:
    """Qualified name -> {kind, signature, docstring} for every def and class."""
    found: dict[str, dict[str, str]] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, prefix)
                continue
            name = f"{prefix}{child.name}"
            is_class = isinstance(child, ast.ClassDef)
            signature = ""
            if not is_class and child.body:
                # why: from the `def` line to the line before the body starts, so
                # a signature spanning five lines of arguments is captured whole
                # and the decorators above it are left out.
                span = lines[child.lineno - 1 : child.body[0].lineno - 1]
                # why: comments are not AST nodes, so `body[0].lineno` sits below
                # any comment block between the signature and the first
                # statement, and those lines would be pasted into the signature.
                signature = " ".join(
                    line.strip() for line in span if not line.strip().startswith("#")
                ).strip()
            found[name] = {
                "kind": "class" if is_class else "function",
                "signature": signature,
                "docstring": first_line(ast.get_docstring(child)),
            }
            walk(child, f"{name}.")

    walk(tree, "")
    return found


def build_header(
    chunk: dict[str, Any],
    module_doc: str,
    imports: list[str],
    defs: dict[str, dict[str, str]],
    encoder: Any,
    header_budget: int,
    imports_budget: int,
) -> str:
    """The `File: / Module: / Class: / Signature: / Imports:` block."""
    parts: list[tuple[str, str]] = [("File", chunk["file_path"])]
    if module_doc:
        parts.append(("Module", module_doc))

    own = defs.get(chunk["qualified_name"], {})
    if own.get("kind") == "class":
        class_name, class_doc = chunk["qualified_name"], own.get("docstring", "")
    else:
        class_name = chunk["parent_name"] if chunk["parent_name"] in defs else ""
        class_doc = defs.get(class_name, {}).get("docstring", "") if class_name else ""
    if class_name:
        parts.append(("Class", f"{class_name} — {class_doc}" if class_doc else class_name))
    if own.get("signature"):
        parts.append(("Signature", own["signature"]))

    kept: list[str] = []
    used = 0
    for name in imports:
        cost = len(encoder.encode_ordinary(name)) + 2
        if used + cost > imports_budget:
            kept.append("...")
            break
        kept.append(name)
        used += cost
    if kept:
        parts.append(("Imports", ", ".join(kept)))

    def render(items: list[tuple[str, str]]) -> str:
        return "\n".join(f"{label}: {value}" for label, value in items)

    for droppable in DROP_ORDER:
        if len(encoder.encode_ordinary(render(parts))) <= header_budget:
            break
        parts = [p for p in parts if p[0] != droppable]

    header = render(parts)
    # why: dropping whole lines is not enough to enforce the cap. A single
    # 900-token signature, or a docstring whose "first line" is a paragraph with
    # no newline, survives every drop and blows the budget on its own. Truncating
    # by token is the only thing that makes the limit a limit.
    ids = encoder.encode_ordinary(header)
    if len(ids) > header_budget:
        header = encoder.decode(ids[:header_budget]) + " …"
    return header


def enrich_file(path: str, chunks: list[dict[str, Any]], encoder: Any) -> Iterator[dict[str, Any]]:
    """Add `enriched_text` to every chunk of one file."""
    source = (CLONE_DIR / path).read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
        module_doc = first_line(ast.get_docstring(tree))
        imports = render_imports(tree)
        defs = definitions(tree, lines)
    except (SyntaxError, ValueError, RecursionError):
        # why: enrichment is additive. A file that will not parse still gets its
        # chunks, with a header holding the only fact we are sure of.
        module_doc, imports, defs = "", [], {}

    for chunk in chunks:
        header = build_header(
            chunk, module_doc, imports, defs, encoder,
            settings.enrich_header_max_tokens, settings.enrich_imports_max_tokens,
        )
        enriched = f"{header}\n{SEPARATOR}\n{chunk['text']}"
        yield {
            **chunk,
            "enriched_text": enriched,
            "n_tokens_enriched": len(encoder.encode_ordinary(enriched)),
        }


SCHEMA = pa.schema(
    list(AST_SCHEMA)
    + [pa.field("enriched_text", pa.string()), pa.field("n_tokens_enriched", pa.int64())]
)


@app.command("build-enriched-chunks")
def build_enriched_chunks() -> None:
    """Read the AST chunks, add headers, write the enriched table."""
    chunks = pq.read_table(AST_CHUNKS_PARQUET).to_pylist()
    by_file: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk["file_path"], []).append(chunk)
    encoder = tiktoken.get_encoding(ENCODING)
    typer.echo(f"enriching {len(chunks):,} chunks across {len(by_file):,} files")

    rows: list[dict[str, Any]] = []
    for done, (path, file_chunks) in enumerate(sorted(by_file.items()), start=1):
        rows.extend(enrich_file(path, file_chunks, encoder))
        if done % 300 == 0:
            typer.echo(f"  {done}/{len(by_file)} files")

    rows.sort(key=lambda r: (r["file_path"], r["start_line"], r["part_index"]))
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), ENRICHED_CHUNKS_PARQUET)
    _report(rows, encoder)


def _report(rows: list[dict[str, Any]], encoder: Any) -> None:
    overhead = sorted(r["n_tokens_enriched"] - r["n_tokens"] for r in rows)
    with_class = sum(1 for r in rows if "\nClass: " in r["enriched_text"])
    with_sig = sum(1 for r in rows if "\nSignature: " in r["enriched_text"])
    with_mod = sum(1 for r in rows if "\nModule: " in r["enriched_text"])
    with_imports = sum(1 for r in rows if "\nImports: " in r["enriched_text"])
    typer.echo(f"\nWrote {len(rows):,} chunks to {ENRICHED_CHUNKS_PARQUET}")
    typer.echo(f"\nheader tokens      : p50 {overhead[len(overhead) // 2]}, "
               f"p95 {overhead[int(len(overhead) * 0.95)]}, max {overhead[-1]}")
    typer.echo(f"chunks with Module : {with_mod:,} ({with_mod / len(rows):.0%})")
    typer.echo(f"chunks with Class  : {with_class:,} ({with_class / len(rows):.0%})")
    typer.echo(f"chunks with Sig    : {with_sig:,} ({with_sig / len(rows):.0%})")
    typer.echo(f"chunks with Imports: {with_imports:,} ({with_imports / len(rows):.0%})")


@app.command("show")
def show(n: int = typer.Option(3, help="How many enriched chunks to print in full.")) -> None:
    """Print enriched chunks so a human can read the headers."""
    rows = pq.read_table(ENRICHED_CHUNKS_PARQUET).to_pylist()
    picks = [r for r in rows if r["node_type"] == "function_definition" and r["parent_name"]]
    for row in picks[:: max(1, len(picks) // n)][:n]:
        typer.echo("=" * 78)
        typer.echo(row["enriched_text"][:1400])
        typer.echo("")


if __name__ == "__main__":
    app()
