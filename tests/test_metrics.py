"""Hand-computed metric values. If these drift, every result is meaningless."""

from math import log2

import pytest

from coderag.eval.harness import RandomRetriever, bootstrap_ci
from coderag.eval.metrics import (
    RetrievedChunk,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    score_query,
)


def chunk(path: str, *names: str) -> RetrievedChunk:
    return RetrievedChunk(f"{path}:{'+'.join(names)}", path, 0.0, names)


def ranked(*groups: str) -> list[set[str]]:
    """"a", "b c" -> [{'a'}, {'b', 'c'}] — one set per rank."""
    return [set(g.split()) for g in groups]


# --- recall ------------------------------------------------------------------


@pytest.mark.parametrize(
    "results, relevant, k, expected",
    [
        (ranked("a", "b", "c"), {"a"}, 3, 1.0),
        (ranked("a", "b", "c"), {"a", "d"}, 3, 0.5),  # one of two found
        (ranked("a", "b", "c"), {"d"}, 3, 0.0),
        (ranked("a", "b", "c"), {"c"}, 2, 0.0),  # outside k
        (ranked("a", "b", "c"), {"a", "b", "c"}, 1, 1 / 3),  # capped by |relevant|
        ([], {"a"}, 10, 0.0),
        (ranked("a"), set(), 10, 0.0),  # no ground truth is not a free win
    ],
)
def test_recall_at_k(results, relevant, k, expected):
    assert recall_at_k(results, relevant, k) == pytest.approx(expected)


def test_recall_ignores_duplicate_retrievals():
    # Returning the same relevant chunk three times finds one thing, not three.
    assert recall_at_k(ranked("a", "a", "a"), {"a", "b"}, 3) == pytest.approx(0.5)


def test_one_chunk_covering_several_relevant_symbols():
    # A wide window can satisfy the whole ground truth at rank 1. This is the
    # naive chunker's real advantage and the metric must not hide it.
    assert recall_at_k(ranked("a b c"), {"a", "b", "c"}, 1) == pytest.approx(1.0)


# --- MRR ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "results, relevant, expected",
    [
        (ranked("a", "b"), {"a"}, 1.0),
        (ranked("b", "a"), {"a"}, 0.5),
        (ranked("b", "c", "a"), {"a"}, 1 / 3),
        (ranked("b", "c"), {"a"}, 0.0),
        (ranked("b", "a", "c"), {"a", "c"}, 0.5),  # first hit only
        (ranked("x y", "a"), {"y"}, 1.0),  # any symbol in the set counts
    ],
)
def test_mrr_at_k(results, relevant, expected):
    assert mrr_at_k(results, relevant, 10) == pytest.approx(expected)


def test_mrr_respects_the_cutoff():
    results = ranked(*[f"x{i}" for i in range(10)], "a")
    assert mrr_at_k(results, {"a"}, 10) == 0.0
    assert mrr_at_k(results, {"a"}, 11) == pytest.approx(1 / 11)


# --- nDCG --------------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(ranked("a", "b"), {"a", "b"}, 10) == pytest.approx(1.0)


def test_ndcg_single_hit_at_rank_two():
    # gain = 1/log2(3); ideal = 1/log2(2) = 1
    assert ndcg_at_k(ranked("x", "a"), {"a"}, 10) == pytest.approx(1 / log2(3))


def test_ndcg_two_hits_at_ranks_one_and_three():
    gain = 1 / log2(2) + 1 / log2(4)
    ideal = 1 / log2(2) + 1 / log2(3)
    assert ndcg_at_k(ranked("a", "x", "b"), {"a", "b"}, 10) == pytest.approx(gain / ideal)


def test_ndcg_ideal_is_capped_at_k():
    # 30 relevant items but k=10: a perfect top-10 must still score 1.0.
    relevant = {f"r{i}" for i in range(30)}
    assert ndcg_at_k(ranked(*[f"r{i}" for i in range(10)]), relevant, 10) == pytest.approx(1.0)


def test_ndcg_is_zero_without_hits():
    assert ndcg_at_k(ranked("x", "y"), {"a"}, 10) == 0.0


# --- symbol identity ---------------------------------------------------------


def test_symbol_keys_are_file_qualified():
    a = chunk("pandas/core/frame.py", "df")
    b = chunk("pandas/core/series.py", "df")
    assert a.symbol_keys.isdisjoint(b.symbol_keys), "same name, two files, no collision"


def test_symbol_keys_cover_every_symbol_in_the_chunk():
    window = chunk("f.py", "One.a", "One.b")
    assert window.symbol_keys == {"f.py::One.a", "f.py::One.b"}


def test_score_query_separates_file_and_symbol_levels():
    # Right file, wrong symbol: file-level hit, symbol-level miss.
    chunks = [chunk("pandas/core/frame.py", "DataFrame.other")]
    scores = score_query(
        chunks,
        gt_files={"pandas/core/frame.py"},
        gt_symbol_keys={"pandas/core/frame.py::DataFrame.merge"},
        k_values=[1, 10],
    )
    assert scores["recall@1_file"] == pytest.approx(1.0)
    assert scores["recall@1_symbol"] == pytest.approx(0.0)


def test_score_query_emits_every_requested_k():
    scores = score_query([], set(), set(), [1, 5, 20])
    for level in ("file", "symbol"):
        for k in (1, 5, 20):
            assert f"recall@{k}_{level}" in scores
        assert f"mrr@10_{level}" in scores
        assert f"ndcg@10_{level}" in scores


# --- bootstrap and floor -----------------------------------------------------


def test_bootstrap_brackets_the_mean():
    values = [0.0] * 50 + [1.0] * 50
    mean, low, high = bootstrap_ci(values, resamples=500)
    assert mean == pytest.approx(0.5)
    assert low < mean < high
    assert 0.0 <= low and high <= 1.0


def test_bootstrap_of_a_constant_has_no_width():
    mean, low, high = bootstrap_ci([0.4] * 20, resamples=200)
    assert (mean, low, high) == pytest.approx((0.4, 0.4, 0.4))


def test_bootstrap_of_empty_is_zero():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_random_retriever_is_seeded_and_bounded():
    pool = [chunk(f"f{i}.py", f"sym{i}") for i in range(50)]
    first = RandomRetriever(pool, seed=7)("query", 10)
    second = RandomRetriever(pool, seed=7)("query", 10)
    assert first == second, "same seed must reproduce the floor"
    assert len(first) == 10
    assert len({c.chunk_id for c in first}) == 10, "must not return duplicates"


def test_random_retriever_handles_k_larger_than_pool():
    assert len(RandomRetriever([chunk("a.py", "x")])("query", 10)) == 1
