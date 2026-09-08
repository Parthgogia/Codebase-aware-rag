"""Retrieval metrics: pure functions over one query's ranked result list.

Nothing here touches disk, config or a retriever. `harness.py` does the
orchestration; this module is only arithmetic, so it can be unit-tested against
hand-written rankings.
"""

from dataclasses import dataclass, field
from math import log2

# why: relevance is binary here. A chunk either contains code the fixing PR
# touched or it does not; there is no graded judgement to average over.
RELEVANT = 1.0


@dataclass(frozen=True)
class RetrievedChunk:
    """One result from a retriever, at one rank.

    A chunk covers one or more symbols. An AST chunk covers exactly one; a
    fixed-size window covers whatever functions its line range happens to
    overlap, which is the whole point of comparing the two.
    """

    chunk_id: str
    file_path: str
    score: float = 0.0
    # why: a tuple of names rather than a single `qualified_name`. Scoring a
    # naive window by one arbitrarily-chosen symbol would understate it, and
    # picking its "main" symbol is a judgement the chunker cannot honestly make.
    symbols: tuple[str, ...] = field(default_factory=tuple)

    @property
    def symbol_keys(self) -> set[str]:
        """File-qualified symbol identities, the symbol-level join keys."""
        # why: a bare qualified name is not unique across the repo -- `df` and
        # `test_series` live in several files each -- so matching on the name
        # alone would count a right-name-wrong-file chunk as a hit.
        return {f"{self.file_path}::{name}" for name in self.symbols}


def recall_at_k(ranked: list[set[str]], relevant: set[str], k: int) -> float:
    """Fraction of the relevant set covered by the top k results.

    why: true recall, not hit-rate. An issue whose fix touched three symbols is
    only fully answered by finding all three, and a system that reliably finds
    one of three is materially worse than one that finds them all. Note the
    consequence: with three relevant items and one symbol per chunk, Recall@1
    can never exceed 0.33.
    """
    if not relevant:
        return 0.0
    found: set[str] = set()
    for keys in ranked[:k]:
        found |= keys
    return len(found & relevant) / len(relevant)


def mrr_at_k(ranked: list[set[str]], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant result, or 0 if none in top k."""
    for rank, keys in enumerate(ranked[:k], start=1):
        if keys & relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[set[str]], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not relevant:
        return 0.0
    gain = sum(
        RELEVANT / log2(rank + 1)
        for rank, keys in enumerate(ranked[:k], start=1)
        if keys & relevant
    )
    # why: the ideal ranking is every relevant item first, but no more of them
    # than fit in k. Without that cap a query with 30 relevant symbols could
    # never score 1.0 at k=10 however perfect the ranking was.
    ideal = sum(RELEVANT / log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0


def score_query(
    chunks: list[RetrievedChunk],
    gt_files: set[str],
    gt_symbol_keys: set[str],
    k_values: list[int],
) -> dict[str, float]:
    """Every metric for one query, at both file and symbol granularity."""
    levels = {
        "file": ([{c.file_path} for c in chunks], gt_files),
        "symbol": ([c.symbol_keys for c in chunks], gt_symbol_keys),
    }
    scores: dict[str, float] = {}
    for level, (ranked, relevant) in levels.items():
        for k in k_values:
            scores[f"recall@{k}_{level}"] = recall_at_k(ranked, relevant, k)
        scores[f"mrr@10_{level}"] = mrr_at_k(ranked, relevant, 10)
        scores[f"ndcg@10_{level}"] = ndcg_at_k(ranked, relevant, 10)
    return scores
