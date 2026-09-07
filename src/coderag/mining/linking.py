"""Link each cached issue to the merged pull request that fixed it.

    uv run python -m coderag.mining.linking link-issues
    uv run python -m coderag.mining.linking link-issues --limit 500
    uv run python -m coderag.mining.linking link-issues --with-timeline

PR-first: scan closed pull requests and read GitHub's own auto-close keywords
out of them. The per-issue timeline crawl this replaced cost ~50x more requests
and, as measured in SUMMARY.md, produced mostly coincidental links. It is still
available behind --with-timeline so that comparison stays reproducible.
"""

import json
import re
from collections import Counter, defaultdict
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from coderag.config import settings
from coderag.mining import github_api
from coderag.paths import (
    INTERIM_DIR,
    ISSUES_DIR,
    PULL_FILES_DIR,
    PULLS_DIR,
    TIMELINES_DIR,
)
from coderag.repo import pinned_commit_date

app = typer.Typer(add_completion=False, no_args_is_help=True)

LINKS_PARQUET = INTERIM_DIR / "issue_pr_links.parquet"
PROGRESS_EVERY = 2000

# Link methods, strongest evidence first.
PR_CLOSES = "pr_closing_keyword"
CROSS_REFERENCED = "timeline_crossref"
BODY_REGEX = "body_regex"

# why: GitHub's auto-close vocabulary, followed by any of the three reference
# forms it accepts. `GH-1234` is pandas' house style and the full issue URL is
# common in PRs opened from the web UI; matching only `#1234` silently dropped
# real links (verified: "closes GH-9053", "closes https://.../issues/15275").
CLOSING_KEYWORD = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*"
    r"(?:#|GH-?|https?://github\.com/[\w.-]+/[\w.-]+/issues/)(\d+)\b",
    re.IGNORECASE,
)
ISSUE_REF = re.compile(r"#(\d+)")


@app.callback()
def main() -> None:
    """Link cached issues to their fixing pull requests."""
    # why: an explicit callback keeps typer from collapsing the subcommand name.


def _repo_url(suffix: str) -> str:
    return f"{github_api.API_ROOT}/repos/{settings.repo_owner}/{settings.repo_name}/{suffix}"


def _closing_refs(pull: dict[str, Any]) -> set[int]:
    """Issue numbers this PR asked GitHub to close, from its title and body."""
    text = f"{pull.get('title') or ''}\n{pull.get('body') or ''}"
    return {int(ref) for ref in CLOSING_KEYWORD.findall(text)}


def _pull_files(number: int) -> list[dict[str, Any]]:
    path = PULL_FILES_DIR / f"{number}.json"
    return github_api.cached_list(path, _repo_url(f"pulls/{number}/files"))


def _eligible_issues(closed_after: str) -> dict[int, dict[str, Any]]:
    """Cached issues closed after the cut-off, keyed by number."""
    eligible: dict[int, dict[str, Any]] = {}
    for path in ISSUES_DIR.glob("*.json"):
        issue = json.loads(path.read_text(encoding="utf-8"))
        # why: string compare on ISO-8601 UTC timestamps is a correct ordering
        # and avoids parsing 8,331 dates to answer a yes/no question.
        if (issue.get("closed_at") or "") > closed_after:
            eligible[int(issue["number"])] = issue
    return eligible


# --- PR-first scan -----------------------------------------------------------


def _scan_pulls(limit: int | None) -> tuple[dict[int, list[dict[str, Any]]], Counter]:
    """Walk closed PRs, returning issue number -> merged PRs that close it."""
    by_issue: dict[int, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()
    params = {
        "state": "closed",
        "per_page": github_api.PER_PAGE,
        # why: created/desc is a stable ordering (created_at never changes) and
        # puts recent PRs first, so a --limit smoke test sees useful data.
        "sort": "created",
        "direction": "desc",
    }
    for batch in github_api.paginate(_repo_url("pulls"), params):
        for pull in batch:
            stats["scanned"] += 1
            # why: the list endpoint omits `merged`, but merged_at is present and
            # is only ever set on merge, so it is the reliable test here.
            if not pull.get("merged_at"):
                continue
            stats["merged"] += 1
            refs = _closing_refs(pull)
            if not refs:
                continue
            stats["with_closing_ref"] += 1
            github_api.write_json(PULLS_DIR / f"{pull['number']}.json", pull)
            for issue_number in refs:
                by_issue[issue_number].append(pull)
        if stats["scanned"] % PROGRESS_EVERY < github_api.PER_PAGE:
            typer.echo(f"  scanned {stats['scanned']:,} PRs, {len(by_issue):,} issues linked")
        if limit is not None and stats["scanned"] >= limit:
            break
    return by_issue, stats


def _row(issue_number: int, pull: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "issue_number": issue_number,
        "pr_number": int(pull["number"]),
        "link_method": method,
        "merge_commit_sha": str(pull.get("merge_commit_sha") or ""),
        "merged_at": str(pull.get("merged_at") or ""),
        "files_changed": len(_pull_files(int(pull["number"]))),
    }


# --- optional timeline pass (weak tiers, kept for comparison) ----------------


def _pull(number: int) -> dict[str, Any] | None:
    """The PR object, or None when this number is not a pull request."""
    payload = github_api.cached_object(PULLS_DIR / f"{number}.json", _repo_url(f"pulls/{number}"))
    # why: the 404 body is cached verbatim too, so a regex candidate that turns
    # out to be a plain issue is never re-requested on a later run.
    return payload if "number" in payload else None


def _weak_rows(issue: dict[str, Any], already: set[int]) -> list[dict[str, Any]]:
    """Cross-reference and bare-`#123` links, for issues the PR scan missed."""
    number = int(issue["number"])
    events = github_api.cached_list(
        TIMELINES_DIR / f"{number}.json", _repo_url(f"issues/{number}/timeline")
    )
    candidates: dict[int, str] = {}
    for event in events:
        if event.get("event") == "cross-referenced":
            source = (event.get("source") or {}).get("issue") or {}
            if "pull_request" in source and "number" in source:
                candidates[int(source["number"])] = CROSS_REFERENCED
    texts = [issue.get("body") or ""]
    texts += [e.get("body") or "" for e in events if e.get("event") == "commented"]
    for ref in {int(r) for t in texts for r in ISSUE_REF.findall(t)} - {number}:
        if ref not in candidates and not (ISSUES_DIR / f"{ref}.json").exists():
            candidates[ref] = BODY_REGEX

    rows = []
    for pr_number, method in sorted(candidates.items()):
        if pr_number in already:
            continue
        pull = _pull(pr_number)
        if pull is not None and pull.get("merged"):
            rows.append(_row(number, pull, method))
    return rows


# --- command -----------------------------------------------------------------


SCHEMA = pa.schema(
    [
        pa.field("issue_number", pa.int64()),
        pa.field("pr_number", pa.int64()),
        pa.field("link_method", pa.string()),
        pa.field("merge_commit_sha", pa.string()),
        pa.field("merged_at", pa.string()),
        pa.field("files_changed", pa.int64()),
    ]
)


@app.command("link-issues")
def link_issues(
    closed_after: str = typer.Option(
        None, help="ISO date. Defaults to the pinned commit's date; earlier issues leak."
    ),
    limit: int = typer.Option(None, help="Stop after scanning N pull requests. Smoke tests."),
    with_timeline: bool = typer.Option(
        False, help="Also mine per-issue timelines for the weak crossref/regex tiers."
    ),
) -> None:
    """Scan merged PRs for closing keywords and write the issue->PR link table."""
    for directory in (PULLS_DIR, PULL_FILES_DIR, TIMELINES_DIR, INTERIM_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cutoff = closed_after or pinned_commit_date()
    cached = len(list(ISSUES_DIR.glob("*.json")))
    eligible = _eligible_issues(cutoff)
    if not eligible:
        raise typer.BadParameter(f"no cached issues closed after {cutoff}; run mine-issues first")
    typer.echo(f"cut-off {cutoff}: {len(eligible):,} eligible issues of {cached:,} cached")

    by_issue, stats = _scan_pulls(limit)
    rows: list[dict[str, Any]] = []
    link_counts: dict[int, int] = {}
    for number in sorted(eligible):
        issue_rows = [_row(number, p, PR_CLOSES) for p in by_issue.get(number, [])]
        if with_timeline:
            issue_rows += _weak_rows(eligible[number], {r["pr_number"] for r in issue_rows})
        rows.extend(issue_rows)
        link_counts[number] = len(issue_rows)

    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), LINKS_PARQUET)
    _report(link_counts, rows, stats)


def _report(link_counts: dict[int, int], rows: list[dict[str, Any]], stats: Counter) -> None:
    counts = Counter(link_counts.values())
    multiple = sum(n for value, n in counts.items() if value > 1)
    total = len(link_counts)
    typer.echo(
        f"\nscanned {stats['scanned']:,} closed PRs "
        f"({stats['merged']:,} merged, {stats['with_closing_ref']:,} with a closing ref)"
    )
    typer.echo(f"Wrote {len(rows):,} links to {LINKS_PARQUET}")
    typer.echo(f"\neligible issues        : {total:,}")
    typer.echo(f"  exactly one merged PR: {counts[1]:,} ({counts[1] / total:.1%})")
    typer.echo(f"  no merged PR         : {counts[0]:,} ({counts[0] / total:.1%})")
    typer.echo(f"  more than one        : {multiple:,} ({multiple / total:.1%})")
    typer.echo("\nlinks by method:")
    for method, n in Counter(r["link_method"] for r in rows).most_common():
        typer.echo(f"  {method:<20}{n:>8,}")


if __name__ == "__main__":
    app()
