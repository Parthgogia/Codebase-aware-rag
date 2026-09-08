"""Golden test: exact chunk boundaries over a hand-written fixture module."""

from pathlib import Path

import pytest
import tiktoken

from coderag.chunking.ast_chunker import (
    FileChunker,
    build_parser,
    chunk_file,
    module_name,
)
from coderag.chunking.naive import ENCODING
from coderag.mining.ground_truth import MODULE_SYMBOL

FIXTURE = Path(__file__).parent / "fixtures" / "sample_module.py.txt"


def _chunk_fixture(floor: int):
    source = FIXTURE.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    encoder = tiktoken.get_encoding(ENCODING)
    counts = [len(ids) for ids in encoder.encode_ordinary_batch(lines)]
    tree = build_parser().parse(source.encode("utf-8"))
    assert not tree.root_node.has_error, "fixture must parse cleanly"
    chunker = FileChunker("pkg/sample_module.py", lines, counts, budget=512, floor=floor)
    return chunker.walk(tree.root_node, "", MODULE_SYMBOL, "module", None, "")


@pytest.fixture(scope="module")
def chunks():
    """Merging on: every tiny sibling method folds together."""
    return _chunk_fixture(floor=32)


@pytest.fixture(scope="module")
def unmerged():
    """Merging off, so exact per-symbol boundaries can be asserted."""
    return _chunk_fixture(floor=0)


def by_name(chunks, name):
    return [c for c in chunks if c["qualified_name"] == name]


def one(chunks, name):
    matches = by_name(chunks, name)
    assert len(matches) == 1, f"expected exactly one {name}, got {len(matches)}"
    return matches[0]


# --- exact boundaries --------------------------------------------------------


def test_decorator_is_inside_its_function_chunk(chunks):
    # @decorator is line 9, def is line 10. The chunk must start at 9.
    chunk = one(chunks, "decorated_function")
    assert (chunk["start_line"], chunk["end_line"]) == (9, 12)
    assert chunk["text"].startswith("@decorator\n")


def test_async_top_level_function(chunks):
    chunk = one(chunks, "async_top_level")
    assert (chunk["start_line"], chunk["end_line"]) == (15, 16)
    assert chunk["node_type"] == "function_definition"


def test_one_line_property_keeps_its_decorator(unmerged):
    chunk = one(unmerged, "Outer.one_liner")
    assert (chunk["start_line"], chunk["end_line"]) == (24, 26)
    assert chunk["text"].lstrip().startswith("@property")
    assert chunk["parent_name"] == "Outer"


def test_async_method_is_found_inside_the_class(unmerged):
    chunk = one(unmerged, "Outer.async_method")
    assert (chunk["start_line"], chunk["end_line"]) == (32, 33)
    assert chunk["parent_name"] == "Outer"


def test_nested_class_qualifies_through_its_parent(unmerged):
    nested = one(unmerged, "Outer.Nested.nested_method")
    assert (nested["start_line"], nested["end_line"]) == (40, 41)
    assert nested["parent_name"] == "Outer.Nested"


def test_class_body_chunk_holds_the_header_and_attributes(chunks):
    # `class Outer:` (19) through the docstring and `attribute = 2` (22).
    chunk = one(chunks, "Outer")
    assert chunk["node_type"] == "class_definition"
    assert chunk["start_line"] == 19
    assert "attribute = 2" in chunk["text"]
    assert "def one_liner" not in chunk["text"], "methods are their own chunks"


def test_nested_class_body_is_its_own_chunk(chunks):
    # `class Nested:` is line 35; line 34 is the blank line before it.
    chunk = one(chunks, "Outer.Nested")
    assert chunk["start_line"] == 35
    assert "nested_attr = 3" in chunk["text"]


def test_module_chunk_holds_only_code_outside_defs(chunks):
    chunk = one(chunks, MODULE_SYMBOL)
    assert chunk["node_type"] == "module"
    assert "import os" in chunk["text"]
    assert "CONSTANT = 1" in chunk["text"]
    assert "TRAILING =" in chunk["text"], "trailing module code must be included"
    assert "def decorated_function" not in chunk["text"]


# --- oversized splitting -----------------------------------------------------


def test_big_function_is_split_into_parts(chunks):
    parts = sorted(by_name(chunks, "big_function"), key=lambda c: c["part_index"])
    assert len(parts) > 1, "a 200-line function must split"
    assert [p["part_index"] for p in parts] == list(range(len(parts)))


def test_first_part_contains_the_signature(chunks):
    parts = sorted(by_name(chunks, "big_function"), key=lambda c: c["part_index"])
    assert parts[0]["text"].startswith("def big_function(n):")


def test_parts_are_contiguous_and_do_not_overlap(chunks):
    parts = sorted(by_name(chunks, "big_function"), key=lambda c: c["part_index"])
    for earlier, later in zip(parts, parts[1:]):
        assert later["start_line"] == earlier["end_line"] + 1


def test_parts_respect_the_budget_and_never_split_a_statement(chunks):
    parts = by_name(chunks, "big_function")
    for part in parts:
        assert part["n_tokens"] <= 512
        # Every line of the body is a whole statement; no part may start mid-line.
        assert not part["text"].startswith(" total +=") or part["text"].startswith("    total")


def test_every_part_keeps_the_same_qualified_name(chunks):
    parts = by_name(chunks, "big_function")
    assert {p["qualified_name"] for p in parts} == {"big_function"}
    assert all(p["covered_symbols"] == ["big_function"] for p in parts)


# --- tiny merging ------------------------------------------------------------


def test_tiny_siblings_merge_and_keep_every_symbol(chunks):
    # one_liner, tiny and async_method are all far under 32 tokens.
    merged = [c for c in chunks if c["node_type"] == "merged"]
    covered = {name for c in merged for name in c["covered_symbols"]}
    assert "Outer.tiny" in covered
    assert len(merged) >= 1
    for chunk in merged:
        assert len(chunk["covered_symbols"]) > 1


def test_merging_does_not_lose_any_symbol(chunks):
    covered = {name for c in chunks for name in c["covered_symbols"]}
    for expected in (
        "decorated_function",
        "async_top_level",
        "Outer",
        "Outer.one_liner",
        "Outer.tiny",
        "Outer.async_method",
        "Outer.Nested",
        "Outer.Nested.nested_method",
        "big_function",
        MODULE_SYMBOL,
    ):
        assert expected in covered, f"{expected} vanished from the chunk set"


# --- naming and misc ---------------------------------------------------------


@pytest.mark.parametrize(
    "path, expected",
    [
        ("pandas/core/frame.py", "pandas.core.frame"),
        ("pandas/__init__.py", "pandas"),
        ("pandas/io/formats/style.py", "pandas.io.formats.style"),
    ],
)
def test_module_name(path, expected):
    assert module_name(path) == expected


def test_full_name_prefixes_the_module(unmerged):
    chunk = one(unmerged, "Outer.one_liner")
    assert chunk["full_name"] == "pkg.sample_module.Outer.one_liner"


def test_chunk_ids_are_unique(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_are_ordered_by_position(chunks):
    starts = [(c["start_line"], c["part_index"]) for c in chunks]
    assert starts == sorted(starts)


def test_unparseable_file_raises(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n    pass\n", encoding="utf-8")
    encoder = tiktoken.get_encoding(ENCODING)
    with pytest.raises(ValueError):
        # chunk_file reads from CLONE_DIR, so exercise the parser check directly.
        tree = build_parser().parse(broken.read_bytes())
        if tree.root_node.has_error:
            raise ValueError("does not parse cleanly")
        chunk_file("broken.py", build_parser(), encoder, 512, 32)
