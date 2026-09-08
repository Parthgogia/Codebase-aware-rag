"""Syntax-aware chunker: one chunk per function, method, class body or module.

    uv run python -m coderag.chunking.ast_chunker build-ast-chunks

The naive chunker cuts every 512 tokens regardless of what it is cutting. This
one cuts on syntax boundaries, so a chunk is a thing a developer would name.
"""

from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
import tree_sitter_python as tree_sitter_python_module
import typer
from tree_sitter import Language, Node, Parser

from coderag.chunking.naive import ENCODING, FILES_PARQUET
from coderag.config import settings
from coderag.mining.ground_truth import MODULE_SYMBOL
from coderag.paths import CLONE_DIR, INDEX_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)

AST_CHUNKS_PARQUET = INDEX_DIR / "chunks_ast.parquet"
DEFINITION_TYPES = ("function_definition", "class_definition")


@app.callback()
def main() -> None:
    """Build AST-aligned chunks over the pinned checkout."""
    # why: an explicit callback keeps typer from collapsing the subcommand name.


def build_parser() -> Parser:
    return Parser(Language(tree_sitter_python_module.language()))


def module_name(path: str) -> str:
    """`pandas/core/frame.py` -> `pandas.core.frame`."""
    stem = path[:-3] if path.endswith(".py") else path
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def definition_of(node: Node) -> Node | None:
    """The def/class inside a possibly-decorated definition, else None."""
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        return inner if inner is not None and inner.type in DEFINITION_TYPES else None
    return node if node.type in DEFINITION_TYPES else None


def name_of(target: Node) -> str:
    identifier = target.child_by_field_name("name")
    return identifier.text.decode("utf-8") if identifier is not None else "<anonymous>"


def _span(node: Node) -> tuple[int, int]:
    """1-based inclusive line range. Includes decorators for a decorated def."""
    return node.start_point[0] + 1, node.end_point[0] + 1


class FileChunker:
    """Chunks one parsed file. Holds the per-file state the walk needs."""

    def __init__(self, path: str, lines: list[str], counts: list[int], budget: int, floor: int):
        self.path = path
        self.module = module_name(path)
        self.lines = lines
        self.counts = counts
        self.budget = budget
        self.floor = floor

    def tokens(self, start: int, end: int) -> int:
        return sum(self.counts[start - 1 : end])

    def text(self, ranges: list[tuple[int, int]]) -> str:
        return "".join("".join(self.lines[s - 1 : e]) for s, e in ranges)

    def chunk(
        self,
        ranges: list[tuple[int, int]],
        qualified: str,
        node_type: str,
        parent: str,
        symbols: tuple[str, ...] | None = None,
        part: int = 0,
    ) -> dict[str, Any]:
        start = min(s for s, _ in ranges)
        end = max(e for _, e in ranges)
        suffix = f"#{part}" if part else ""
        return {
            "chunk_id": f"ast:{self.path}:{start}-{end}{suffix}",
            "file_path": self.path,
            "start_line": start,
            "end_line": end,
            "text": self.text(ranges),
            "covered_symbols": list(symbols or (qualified,)),
            "n_tokens": sum(self.tokens(s, e) for s, e in ranges),
            "qualified_name": qualified,
            # why: `module.Class.method` is what the step asks to record, but the
            # ground truth from Step 5 keys on the in-file name and the join is
            # (file_path, qualified_name). Storing both keeps the scoring join
            # working without losing the fully-qualified form.
            "full_name": f"{self.module}.{qualified}",
            "node_type": node_type,
            "parent_name": parent,
            "part_index": part,
        }

    # --- oversized functions -------------------------------------------------

    def split_function(self, node: Node, target: Node, qualified: str, parent: str) -> list[dict]:
        """One chunk, or several split on top-level statement boundaries."""
        start, end = _span(node)
        if self.tokens(start, end) <= self.budget:
            return [self.chunk([(start, end)], qualified, target.type, parent)]

        body = target.child_by_field_name("body")
        if body is None:
            return [self.chunk([(start, end)], qualified, target.type, parent)]
        # why: the first unit is the decorators and signature, so part 0 always
        # says what the function is. Splitting that off would leave every later
        # part anonymous text with no `def` line in it.
        units = [(start, body.start_point[0])]
        units += [_span(statement) for statement in body.named_children]

        parts: list[tuple[int, int]] = []
        current = units[0]
        used = self.tokens(*current)
        for unit in units[1:]:
            size = self.tokens(*unit)
            # why: never mid-statement. A statement larger than the budget on its
            # own becomes an oversized part rather than being cut in half.
            if used + size > self.budget and used > 0:
                parts.append(current)
                current, used = unit, size
            else:
                current = (current[0], unit[1])
                used += size
        parts.append((current[0], end))
        return [
            self.chunk([span], qualified, target.type, parent, part=index)
            for index, span in enumerate(parts)
        ]

    # --- tiny siblings -------------------------------------------------------

    def merge_tiny(self, chunks: list[dict], parent: str) -> list[dict]:
        """Fold runs of consecutive under-sized siblings into one chunk."""
        merged: list[dict] = []
        run: list[dict] = []

        def flush() -> None:
            if not run:
                return
            if len(run) == 1:
                merged.append(run[0])
            else:
                names = tuple(name for c in run for name in c["covered_symbols"])
                merged.append(
                    self.chunk(
                        [(run[0]["start_line"], run[-1]["end_line"])],
                        # why: the first member names the chunk so it stays
                        # readable; `covered_symbols` carries all of them and is
                        # what scoring actually joins on.
                        run[0]["qualified_name"],
                        "merged",
                        parent,
                        symbols=names,
                    )
                )
            run.clear()

        for chunk in chunks:
            # why: direct methods of *this* class only. `chunks` is the flattened
            # result of the recursive walk, so it also holds a nested class's own
            # body and methods; merging those in would glue `Outer.tiny` to
            # `Outer.Nested.nested_method`, which are not siblings at all. Class
            # bodies stay separate too -- a class header is not a method.
            mergeable = (
                chunk["node_type"] == "function_definition"
                and chunk["parent_name"] == parent
                and chunk["part_index"] == 0
                and chunk["n_tokens"] < self.floor
            )
            if mergeable:
                run.append(chunk)
                continue
            flush()
            merged.append(chunk)
        flush()
        return merged

    # --- the walk ------------------------------------------------------------

    def walk(
        self,
        block: Node,
        prefix: str,
        own_name: str,
        own_type: str,
        header: tuple[int, int] | None,
        parent: str,
    ) -> list[dict]:
        """Chunks for one block: its definitions, plus its own leftover code."""
        own_ranges: list[tuple[int, int]] = [header] if header else []
        children: list[dict] = []

        for child in block.named_children:
            target = definition_of(child)
            if target is None:
                own_ranges.append(_span(child))
                continue
            qualified = f"{prefix}{name_of(target)}"
            if target.type == "class_definition":
                body = target.child_by_field_name("body")
                if body is None:
                    children.append(
                        self.chunk([_span(child)], qualified, target.type, own_name)
                    )
                    continue
                class_header = (child.start_point[0] + 1, body.start_point[0])
                children += self.walk(
                    body, f"{qualified}.", qualified, "class_definition",
                    class_header, own_name,
                )
            else:
                children += self.split_function(child, target, qualified, own_name)

        # why: merging only inside a class, as the step specifies. Module-level
        # helpers are rarely siblings in any meaningful sense, and merging them
        # would glue unrelated top-level functions together.
        if own_type == "class_definition":
            children = self.merge_tiny(children, own_name)

        chunks = children
        if own_ranges:
            chunks = chunks + [
                self.chunk(own_ranges, own_name, own_type, parent)
            ]
        return sorted(chunks, key=lambda c: (c["start_line"], c["part_index"]))


def chunk_file(path: str, parser: Parser, encoder: Any, budget: int, floor: int) -> list[dict]:
    """Every chunk of one file. Raises ValueError if the file will not parse."""
    source = (CLONE_DIR / path).read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines(keepends=True)
    if not lines:
        return []
    tree = parser.parse(source.encode("utf-8"))
    if tree.root_node.has_error:
        raise ValueError(f"{path} does not parse cleanly")
    counts = [len(ids) for ids in encoder.encode_ordinary_batch(lines)]
    chunker = FileChunker(path, lines, counts, budget, floor)
    return chunker.walk(tree.root_node, "", MODULE_SYMBOL, "module", None, "")


SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("start_line", pa.int64()),
        pa.field("end_line", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("covered_symbols", pa.list_(pa.string())),
        pa.field("n_tokens", pa.int64()),
        pa.field("qualified_name", pa.string()),
        pa.field("full_name", pa.string()),
        pa.field("node_type", pa.string()),
        pa.field("parent_name", pa.string()),
        pa.field("part_index", pa.int64()),
    ]
)


def _python_files() -> list[str]:
    rows = pq.read_table(FILES_PARQUET).to_pylist()
    return sorted(r["path"] for r in rows if r["extension"] == ".py")


@app.command("build-ast-chunks")
def build_ast_chunks(
    budget: int = typer.Option(None, help="Max tokens per chunk; defaults to chunk_max_tokens."),
    floor: int = typer.Option(None, help="Merge below this; defaults to chunk_min_tokens."),
) -> None:
    """Chunk every Python file on syntax boundaries and write the parquet."""
    budget = budget or settings.chunk_max_tokens
    floor = floor or settings.chunk_min_tokens
    parser, encoder = build_parser(), tiktoken.get_encoding(ENCODING)
    paths = _python_files()
    typer.echo(f"chunking {len(paths):,} Python files (max {budget}, merge under {floor})")

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for done, path in enumerate(paths, start=1):
        try:
            rows.extend(chunk_file(path, parser, encoder, budget, floor))
        except ValueError:
            skipped.append(path)
        if done % 300 == 0:
            typer.echo(f"  {done}/{len(paths)} files, {len(rows):,} chunks")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), AST_CHUNKS_PARQUET)
    _report(rows, len(paths), skipped)


def _report(rows: list[dict[str, Any]], n_files: int, skipped: list[str]) -> None:
    tokens = sorted(r["n_tokens"] for r in rows)
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["node_type"]] = kinds.get(row["node_type"], 0) + 1
    split = sum(1 for r in rows if r["part_index"] > 0)
    typer.echo(f"\nWrote {len(rows):,} chunks to {AST_CHUNKS_PARQUET}")
    typer.echo(f"\nfiles chunked      : {n_files - len(skipped):,}")
    typer.echo(f"  failed to parse  : {len(skipped):,} {skipped[:3]}")
    typer.echo(f"chunks per file    : mean {len(rows) / max(1, n_files - len(skipped)):.1f}")
    typer.echo(
        f"tokens per chunk   : p50 {tokens[len(tokens) // 2]}, "
        f"p95 {tokens[int(len(tokens) * 0.95)]}, max {tokens[-1]}"
    )
    typer.echo(f"continuation parts : {split:,} (from oversized functions)")
    typer.echo("\nby node type:")
    for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {kind:<22}{count:>8,}")


if __name__ == "__main__":
    app()
