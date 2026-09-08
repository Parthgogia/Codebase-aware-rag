"""Run any retriever over the eval set and produce a results table.

    uv run python -m coderag.eval.harness evaluate-random

A retriever is any callable `(query: str, k: int) -> list[RetrievedChunk]`.
There is deliberately no base class to inherit: a function is the whole
contract, and a plain callable can be a closure, a bound method or a lambda
without anything having to be registered.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pyarrow.parquet as pq
import typer

from coderag.config import settings
from coderag.eval.dataset import QUERIES_PARQUET
from coderag.eval.metrics import RetrievedChunk, score_query
from coderag.mining.ground_truth import GROUND_TRUTH_PARQUET
from coderag.paths import RESULTS_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)

BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE = 95
RETRIEVE_DEPTH = 20


class Retriever(Protocol):
    """Structural type only -- nothing subclasses this."""

    def __call__(self, query: str, k: int) -> list[RetrievedChunk]: ...


@dataclass
class ResultsTable:
    """Metrics for one run, overall and per bucket, with intervals."""

    run_name: str
    n_queries: int
    overall: dict[str, tuple[float, float, float]]
    by_bucket: dict[str, dict[str, tuple[float, float, float]]]
    bucket_sizes: dict[str, int]
    config: dict[str, Any] = field(default_factory=dict)

    def save(self) -> "object":
        """Write results/<run_name>.json and return the path."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{self.run_name}.json"
        payload = {
            "run_name": self.run_name,
            "n_queries": self.n_queries,
            "bucket_sizes": self.bucket_sizes,
            "overall": {k: list(v) for k, v in self.overall.items()},
            "by_bucket": {
                b: {k: list(v) for k, v in m.items()} for b, m in self.by_bucket.items()
            },
            "config": self.config,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def render(self) -> str:
        """A fixed-width table of the headline metrics."""
        keys = [k for k in self.overall if k.startswith(("recall@10", "mrr", "ndcg"))]
        lines = [f"{'metric':<22}{'overall':>22}" + "".join(f"{b:>22}" for b in self.by_bucket)]
        lines.append("-" * (22 + 22 * (1 + len(self.by_bucket))))
        for key in sorted(keys):
            row = f"{key:<22}{_fmt(self.overall[key]):>22}"
            row += "".join(_fmt(self.by_bucket[b].get(key)) .rjust(22) for b in self.by_bucket)
            lines.append(row)
        sizes = ", ".join(f"{b}={n}" for b, n in self.bucket_sizes.items())
        lines.append(f"\nn={self.n_queries} ({sizes}); 95% CI from {BOOTSTRAP_RESAMPLES} resamples")
        return "\n".join(lines)


def _fmt(value: tuple[float, float, float] | None) -> str:
    if value is None:
        return "-"
    return f"{value[0]:.3f} [{value[1]:.3f},{value[2]:.3f}]"


# --- aggregation -------------------------------------------------------------


def bootstrap_ci(values: list[float], resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    """(mean, lower, upper) with a percentile bootstrap over queries."""
    if not values:
        return (0.0, 0.0, 0.0)
    array = np.asarray(values, dtype=float)
    # why: resample whole queries, not individual scores. The unit of
    # uncertainty is "which issues happened to end up in the eval set", which is
    # what a different sample of pandas issues would have changed.
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[picks].mean(axis=1)
    tail = (100 - CONFIDENCE) / 2
    return (float(array.mean()), float(np.percentile(means, tail)),
            float(np.percentile(means, 100 - tail)))


def _aggregate(per_query: list[dict[str, float]]) -> dict[str, tuple[float, float, float]]:
    if not per_query:
        return {}
    return {key: bootstrap_ci([q[key] for q in per_query]) for key in per_query[0]}


def config_snapshot() -> dict[str, Any]:
    """Every setting that could have influenced this run, minus secrets."""
    snapshot = settings.model_dump(mode="json")
    # why: the results file is meant to be committed next to the numbers, and a
    # token in git history is a token that has to be revoked.
    snapshot.pop("github_token", None)
    return snapshot


# --- the harness -------------------------------------------------------------


def load_queries() -> list[dict[str, Any]]:
    return pq.read_table(QUERIES_PARQUET).to_pylist()


def evaluate(
    retriever: Retriever,
    queries: list[dict[str, Any]],
    run_name: str,
    k_values: list[int] | None = None,
    text_field: str = "query_text",
) -> ResultsTable:
    """Score a retriever over the eval set, overall and per bucket."""
    k_values = k_values or settings.eval_k_values
    depth = max(max(k_values), 10)
    per_query: list[dict[str, float]] = []
    buckets: dict[str, list[dict[str, float]]] = {}

    for query in queries:
        chunks = retriever(query[text_field], depth)
        scores = score_query(
            chunks,
            set(query["gt_files"]),
            set(query["gt_symbol_keys"]),
            k_values,
        )
        per_query.append(scores)
        buckets.setdefault(query["bucket"], []).append(scores)

    return ResultsTable(
        run_name=run_name,
        n_queries=len(queries),
        overall=_aggregate(per_query),
        by_bucket={b: _aggregate(rows) for b, rows in sorted(buckets.items())},
        bucket_sizes={b: len(rows) for b, rows in sorted(buckets.items())},
        config=config_snapshot(),
    )


# --- the floor ---------------------------------------------------------------


class RandomRetriever:
    """Samples uniformly from a fixed pool. The score any system must beat."""

    def __init__(self, pool: list[RetrievedChunk], seed: int = 0) -> None:
        self.pool = pool
        self.random = random.Random(seed)

    def __call__(self, query: str, k: int) -> list[RetrievedChunk]:
        return self.random.sample(self.pool, min(k, len(self.pool)))


def ground_truth_pool() -> list[RetrievedChunk]:
    """Every (file, symbol) pair any fix ever touched, as a candidate chunk.

    why: a harder floor than sampling random files. This pool contains only
    symbols that were genuinely changed by some bug fix, so it is the most
    favourable universe a chance retriever could draw from -- if the metric
    still reads ~0 here, it is not rewarding plausibility.
    """
    rows = pq.read_table(GROUND_TRUTH_PARQUET).to_pylist()
    seen = {(r["file_path"], r["qualified_name"]) for r in rows}
    return [
        RetrievedChunk(f"{path}::{name}", path, 0.0, (name,))
        for path, name in sorted(seen)
    ]


@app.callback()
def main() -> None:
    """Evaluate retrievers against the query set."""
    # why: an explicit callback keeps typer from collapsing the subcommand name.


@app.command("evaluate-random")
def evaluate_random(
    run_name: str = typer.Option("random", help="Results are written to results/<name>.json"),
    seed: int = typer.Option(0, help="Sampling seed, so the floor is reproducible."),
) -> None:
    """Run the random-chunk floor and print the results table."""
    queries = load_queries()
    pool = ground_truth_pool()
    typer.echo(f"{len(queries):,} queries, sampling from {len(pool):,} candidate chunks")
    table = evaluate(RandomRetriever(pool, seed), queries, run_name)
    typer.echo("\n" + table.render())
    typer.echo(f"\nwrote {table.save()}")


if __name__ == "__main__":
    app()
