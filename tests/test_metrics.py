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


def chunk(path: str, name: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(f"{path}::{name}", path, name, score)


# --- recall ------------------------------------------------------------------


@pytest.mark.parametrize(
    "ranked, relevant, k, expected",
    [
        (["a", "b", "c"], {"a"}, 3, 1.0),
        (["a", "b", "c"], {"a", "d"}, 3, 0.5),  # one of two found
        (["a", "b", "c"], {"d"}, 3, 0.0),
        (["a", "b", "c"], {"c"}, 2, 0.0),  # outside k
        (["a", "b", "c"], {"a", "b", "c"}, 1, 1 / 3),  # recall@1 is capped by |relevant|
        ([], {"a"}, 10, 0.0),
        (["a"], set(), 10, 0.0),  # no ground truth is not a free win
    ],
)
def test_recall_at_k(ranked, relevant, k, expected):
    assert recall_at_k(ranked, relevant, k) == pytest.approx(expected)


def test_recall_ignores_duplicate_retrievals():
    # Returning the same relevant chunk five times finds one thing, not five.
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == pytest.approx(0.5)


# --- MRR ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "ranked, relevant, expected",
    [
        (["a", "b"], {"a"}, 1.0),
        (["b", "a"], {"a"}, 0.5),
        (["b", "c", "a"], {"a"}, 1 / 3),
        (["b", "c"], {"a"}, 0.0),
        (["b", "a", "c"], {"a", "c"}, 0.5),  # first hit only
    ],
)
def test_mrr_at_k(ranked, relevant, expected):
    assert mrr_at_k(ranked, relevant, 10) == pytest.approx(expected)


def test_mrr_respects_the_cutoff():
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    assert mrr_at_k(ranked, {"a"}, 10) == 0.0
    assert mrr_at_k(ranked, {"a"}, 11) == pytest.approx(1 / 11)


# --- nDCG --------------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_ndcg_single_hit_at_rank_two():
    # gain = 1/log2(3); ideal = 1/log2(2) = 1
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / log2(3))


def test_ndcg_two_hits_at_ranks_one_and_three():
    gain = 1 / log2(2) + 1 / log2(4)
    ideal = 1 / log2(2) + 1 / log2(3)
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10) == pytest.approx(gain / ideal)


def test_ndcg_ideal_is_capped_at_k():
    # 30 relevant items but k=10: a perfect top-10 must still score 1.0.
    relevant = {f"r{i}" for i in range(30)}
    ranked = [f"r{i}" for i in range(10)]
    assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(1.0)


def test_ndcg_is_zero_without_hits():
    assert ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0


# --- symbol identity ---------------------------------------------------------


def test_symbol_key_is_file_qualified():
    a = chunk("pandas/core/frame.py", "df")
    b = chunk("pandas/core/series.py", "df")
    assert a.symbol_key != b.symbol_key, "same name in two files must not collide"


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
    pool = [chunk("a.py", "x")]
    assert len(RandomRetriever(pool)("query", 10)) == 1
