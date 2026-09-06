"""Mine every closed issue of the target repo into a raw on-disk cache.

    uv run python -m coderag.mining.issues mine-issues --since 2023-01-01 --limit 50

The cache is the contract: this is the only module in the pipeline allowed to
talk to the GitHub API for issues. Everything downstream reads
`data/raw/issues/<number>.json` from disk.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests
import typer

from coderag.config import settings
from coderag.paths import ISSUES_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)

API_ROOT = "https://api.github.com"
PER_PAGE = 100
PROGRESS_EVERY = 100
MAX_ATTEMPTS = 5
# why: a margin, not 0. The remaining counter is per-token and other processes
# (or a previous run's in-flight requests) can spend the last unit under us.
LOW_REMAINING = 2


@app.callback()
def main() -> None:
    """Fetch closed GitHub issues into the raw cache."""
    # why: an explicit callback forces typer to keep the subcommand name, so the
    # documented `mine-issues` invocation works instead of being collapsed away.


# --- HTTP --------------------------------------------------------------------


def _headers() -> dict[str, str]:
    """Standard GitHub REST headers, with the token when one is configured."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def _seconds_until_reset(response: requests.Response) -> float:
    """How long to wait before the quota refills, from the response headers."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        # why: secondary rate limits answer with Retry-After and no useful reset
        # header. Honouring it is what keeps GitHub from blocking the token.
        return float(retry_after)
    reset = response.headers.get("X-RateLimit-Reset")
    if reset is None:
        return 60.0
    now = datetime.now(timezone.utc).timestamp()
    return max(0.0, float(reset) - now)


def _sleep_if_exhausted(response: requests.Response) -> None:
    """Block until the quota resets if this response used up the budget."""
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is None or int(remaining) > LOW_REMAINING:
        return
    wait = _seconds_until_reset(response) + 5
    typer.echo(f"  rate limit low ({remaining} left); sleeping {wait:.0f}s")
    time.sleep(wait)


def _get(url: str, params: dict[str, Any]) -> requests.Response:
    """GET with retries that sleep on rate limits instead of hammering."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = requests.get(url, headers=_headers(), params=params, timeout=30)
        if response.status_code == 200:
            _sleep_if_exhausted(response)
            return response
        rate_limited = response.status_code in (403, 429) and (
            response.headers.get("X-RateLimit-Remaining") == "0"
            or "Retry-After" in response.headers
        )
        if rate_limited:
            wait = _seconds_until_reset(response) + 5
            typer.echo(f"  rate limited; sleeping {wait:.0f}s")
            time.sleep(wait)
        elif response.status_code >= 500:
            typer.echo(f"  HTTP {response.status_code}; retry {attempt}")
            time.sleep(2**attempt)
        else:
            response.raise_for_status()
    raise RuntimeError(f"giving up on {url} after {MAX_ATTEMPTS} attempts")


def _issue_pages(since: str | None) -> Iterator[list[dict[str, Any]]]:
    """Yield successive pages of closed issues (which include pull requests)."""
    url = f"{API_ROOT}/repos/{settings.repo_owner}/{settings.repo_name}/issues"
    params: dict[str, Any] = {
        "state": "closed",
        "per_page": PER_PAGE,
        # why: sort by creation ascending. The default (updated, descending)
        # reorders under us while we page, which silently skips and duplicates
        # items across page boundaries on a repo this active.
        "sort": "created",
        "direction": "asc",
    }
    if since:
        params["since"] = since
    while True:
        response = _get(url, params)
        batch = response.json()
        if not batch:
            return
        yield batch
        # why: follow the Link header's rel="next" rather than incrementing a
        # page number. GitHub refuses ?page= past 10,000 items with a 422 and
        # tells you to use its cursor instead; that next URL already carries the
        # cursor plus every filter, so the query params are only sent once.
        next_url = response.links.get("next", {}).get("url")
        if not next_url:
            return
        url, params = next_url, {}


# --- cache -------------------------------------------------------------------


def _cache_path(number: int) -> Path:
    return ISSUES_DIR / f"{number}.json"


def _write_issue(issue: dict[str, Any]) -> None:
    """Write the raw API object verbatim, atomically."""
    path = _cache_path(int(issue["number"]))
    tmp = path.with_suffix(".json.tmp")
    # why: write-then-rename. A kill in the middle of a plain write leaves a
    # truncated file that a later run would happily treat as cached.
    tmp.write_text(json.dumps(issue, indent=2), encoding="utf-8")
    tmp.replace(path)


# --- command -----------------------------------------------------------------


@app.command("mine-issues")
def mine_issues(
    since: str = typer.Option(None, help="ISO date; filters on issue UPDATE time."),
    limit: int = typer.Option(None, help="Stop after this many issues. Smoke tests."),
) -> None:
    """Page through closed issues and cache each non-PR one as raw JSON."""
    ISSUES_DIR.mkdir(parents=True, exist_ok=True)
    if not settings.github_token:
        typer.echo("No GITHUB_TOKEN set: 60 requests/hour instead of 5000.")

    seen = written = skipped = pull_requests = 0
    for batch in _issue_pages(since):
        for item in batch:
            # why: the issues endpoint returns PRs too, distinguished only by
            # this key. Number ranges overlap, so filtering here stops a PR from
            # overwriting the issue cache entry with the same number.
            if "pull_request" in item:
                pull_requests += 1
                continue
            seen += 1
            if _cache_path(int(item["number"])).exists():
                skipped += 1
            else:
                _write_issue(item)
                written += 1
            if seen % PROGRESS_EVERY == 0:
                typer.echo(f"  {seen} issues ({written} new, {skipped} cached)")
            if limit is not None and seen >= limit:
                _report(seen, written, skipped, pull_requests)
                return
    _report(seen, written, skipped, pull_requests)


def _report(seen: int, written: int, skipped: int, pull_requests: int) -> None:
    typer.echo(
        f"\nissues seen      : {seen:,}"
        f"\n  newly cached   : {written:,}"
        f"\n  already cached : {skipped:,}"
        f"\npull requests skipped: {pull_requests:,}"
        f"\ncache: {ISSUES_DIR}"
    )


if __name__ == "__main__":
    app()
