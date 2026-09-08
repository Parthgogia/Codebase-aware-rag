"""The tokenizer table. These inputs are where naive code tokenizers go wrong."""

import pytest

from coderag.retrieval.sparse import KEYWORDS, split_identifier, tokenize_code


@pytest.mark.parametrize(
    "identifier, expected",
    [
        ("_resolve_lookup", ["resolve", "lookup"]),
        ("__init__", ["init"]),
        ("HTTPSConnection", ["HTTPS", "Connection"]),
        ("get_HTTP_response", ["get", "HTTP", "response"]),
        ("x2y", ["x2y"]),
        ("DataFrame", ["Data", "Frame"]),
        ("read_csv", ["read", "csv"]),
        ("NDFrame", ["ND", "Frame"]),
        ("to_HDF5", ["to", "HDF", "5"]),
        ("snake", ["snake"]),
        ("_private", ["private"]),
        ("__dunder__", ["dunder"]),
        ("ALLCAPS", ["ALLCAPS"]),
        ("parse_JSONValue", ["parse", "JSON", "Value"]),
        ("a", ["a"]),
        ("_", []),
        ("value2", ["value2"]),
        ("Value2Thing", ["Value2", "Thing"]),
    ],
)
def test_split_identifier(identifier, expected):
    assert split_identifier(identifier) == expected


def test_https_connection_keeps_the_acronym_whole():
    # The question the step asks: HTTPS must not become HTTP + SConnection.
    tokens = tokenize_code("HTTPSConnection")
    assert tokens == ["httpsconnection", "https", "connection"]


def test_whole_identifier_and_parts_are_both_kept():
    assert tokenize_code("_resolve_lookup") == ["_resolve_lookup", "resolve", "lookup"]


def test_dunder_keeps_the_exact_form():
    # A traceback pastes `__init__` verbatim; the exact term has to be indexed.
    tokens = tokenize_code("__init__")
    assert "__init__" in tokens and "init" in tokens


def test_single_word_identifier_is_not_duplicated():
    # `total` would otherwise be emitted twice, inflating its term frequency.
    assert tokenize_code("total") == ["total"]


def test_keywords_are_dropped():
    tokens = tokenize_code("if self.value is None: return True")
    for dropped in ("if", "is", "none", "return", "true", "self"):
        assert dropped not in tokens
    assert "value" in tokens


def test_keyword_parts_are_dropped_too():
    # `is_none` splits into `is` and `none`, both stopwords; the whole survives.
    tokens = tokenize_code("is_none")
    assert tokens == ["is_none"]


def test_everything_is_lowercased():
    assert all(t == t.lower() for t in tokenize_code("DataFrame.NDFrame HTTPResponse"))


def test_punctuation_and_operators_are_dropped():
    assert tokenize_code("a + b == c[0]  # note!") == ["a", "b", "c", "0", "note"]


def test_dotted_attribute_splits_into_its_parts():
    tokens = tokenize_code("df.read_csv")
    assert tokens == ["df", "read_csv", "read", "csv"]


def test_numbers_survive():
    assert "512" in tokenize_code("chunk size is 512 tokens")


def test_empty_and_symbol_only_text():
    assert tokenize_code("") == []
    assert tokenize_code("!!! ??? ...") == []


def test_realistic_traceback_line():
    tokens = tokenize_code('File "pandas/core/frame.py", line 42, in _reindex_axes')
    for expected in ("pandas", "core", "frame", "py", "_reindex_axes", "reindex", "axes"):
        assert expected in tokens


def test_keyword_set_covers_soft_keywords_and_self():
    for word in ("self", "cls", "return", "lambda", "async", "await"):
        assert word in KEYWORDS
