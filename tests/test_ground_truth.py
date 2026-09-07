"""Hand-built diffs and sources, so the mapping is checked without the network."""

import pytest

from coderag.mining.ground_truth import (
    FILE_SYMBOL,
    MODULE_SYMBOL,
    innermost,
    pre_image_lines,
    symbols,
)

SOURCE = '''\
"""Module docstring."""
import os


TOP_LEVEL = 1


def free_function(a, b):
    """Adds."""
    return a + b


class Outer:
    """A class."""

    attribute = 2

    @property
    def prop(self):
        return self.attribute

    async def method(self, x):
        def inner(y):
            return y * 2

        return inner(x)


class Empty:
    pass
'''


@pytest.fixture
def spans():
    return symbols(SOURCE)


def test_every_symbol_is_found(spans):
    assert [name for name, _, _ in spans] == [
        "free_function",
        "Outer",
        "Outer.prop",
        "Outer.method",
        "Outer.method.inner",
        "Empty",
    ]


@pytest.mark.parametrize(
    "line, expected",
    [
        (1, MODULE_SYMBOL),  # docstring
        (5, MODULE_SYMBOL),  # TOP_LEVEL = 1
        (9, "free_function"),  # its docstring
        (16, "Outer"),  # class attribute, not inside a method
        (18, "Outer.prop"),  # the @property decorator line itself
        (20, "Outer.prop"),  # the return
        (22, "Outer.method"),  # async def
        (24, "Outer.method.inner"),  # innermost wins over method and class
        (26, "Outer.method"),  # back out of inner
        (30, "Empty"),  # `pass`; the class spans 29-30
    ],
)
def test_innermost_symbol(spans, line, expected):
    assert innermost(spans, line) == expected


def test_decorator_line_belongs_to_its_function(spans):
    prop = next(s for s in spans if s[0] == "Outer.prop")
    assert prop[1] == 18, "span must start at the decorator, not the def"


def test_module_level_when_no_symbol_encloses():
    assert innermost([], 42) == MODULE_SYMBOL


# --- diff parsing ------------------------------------------------------------


def test_deletions_use_their_own_pre_image_lines():
    patch = "@@ -10,3 +10,2 @@\n context\n-removed\n context\n"
    assert pre_image_lines(patch) == [11]


def test_insertions_anchor_to_the_line_above():
    patch = "@@ -10,2 +10,4 @@\n context\n+added one\n+added two\n context\n"
    # After one context line the next pre-image line is 11, so both additions
    # anchor to line 10 -- the last line that exists in the indexed tree.
    assert pre_image_lines(patch) == [10, 10]


def test_modification_records_only_the_removed_line():
    patch = "@@ -5,3 +5,3 @@\n a\n-old\n+new\n b\n"
    assert pre_image_lines(patch) == [6, 6]


def test_multiple_hunks_reset_the_counter():
    patch = "@@ -10,2 +10,1 @@\n a\n-x\n@@ -50,2 +49,1 @@\n b\n-y\n"
    assert pre_image_lines(patch) == [11, 51]


def test_hunk_header_without_a_length():
    # A one-line modification: the deletion is line 7, and the replacement
    # anchors to line 7 too, because that is the line it replaces.
    patch = "@@ -7 +7 @@\n-only\n+new\n"
    assert pre_image_lines(patch) == [7, 7]


def test_no_newline_marker_is_not_counted():
    patch = "@@ -1,2 +1,2 @@\n-a\n\\ No newline at end of file\n+b\n"
    assert pre_image_lines(patch) == [1, 1]


def test_insertion_at_the_top_of_a_file_clamps_to_line_one():
    patch = "@@ -1,0 +1,2 @@\n+import os\n+import sys\n"
    assert pre_image_lines(patch) == [1, 1]


def test_end_to_end_mapping_of_a_patch_to_symbols():
    spans = symbols(SOURCE)
    # Touch free_function's return (line 10) and Outer.prop's return (line 20).
    patch = (
        "@@ -9,2 +9,2 @@\n"
        '     """Adds."""\n'
        "-    return a + b\n"
        "+    return b + a\n"
        "@@ -19,2 +19,2 @@\n"
        "     def prop(self):\n"
        "-        return self.attribute\n"
        "+        return self.attribute + 1\n"
    )
    labels = [innermost(spans, line) for line in pre_image_lines(patch)]
    assert labels == ["free_function", "free_function", "Outer.prop", "Outer.prop"]


def test_sentinels_are_distinct():
    assert MODULE_SYMBOL != FILE_SYMBOL
