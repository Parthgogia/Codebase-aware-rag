"""Mine every closed issue of the target repo into a raw on-disk cache.

    uv run python -m coderag.mining.issues mine-issues --since 2023-01-01 --limit 50

The cache is the contract: this is the only module that asks GitHub for issues.
Everything downstream reads `data/raw/issues/<number>.json` from disk.
"""

from pathlib import Path
from typing import Any

import typer

from coderag.config import settings
from coderag.mining import github_api
from coderag.paths import ISSUES_DIR

app = typer.Typer(add_completion=False, no_args_is_help=True)

PROGRESS_EVERY = 100


@app.callback()
def main() -> None:
    """Fetch closed GitHub issues into the raw cache."""
    # why: an explicit callback forces typer to keep the subcommand name, so the
    # documented `mine-issues` invocation works instead of being collapsed away.


def _cache_path(number: int) -> Path:
    return ISSUES_DIR / f"{number}.json"


@app.command("mine-issues")
def mine_issues(
    since: str = typer.Option(None, help="ISO date; filters on issue UPDATE time."),
    limit: int = typer.Option(None, help="Stop after this many issues. Smoke tests."),
) -> None:
    """Page through closed issues and cache each non-PR one as raw JSON."""
    ISSUES_DIR.mkdir(parents=True, exist_ok=True)
    if not settings.github_token:
        typer.echo("No GITHUB_TOKEN set: 60 requests/hour instead of 5000.")

    url = f"{github_api.API_ROOT}/repos/{settings.repo_owner}/{settings.repo_name}/issues"
    params: dict[str, Any] = {
        "state": "closed",
        "per_page": github_api.PER_PAGE,
        # why: sort by creation ascending. The default (updated, descending)
        # reorders under us while we page, which silently skips and duplicates
        # items across page boundaries on a repo this active.
        "sort": "created",
        "direction": "asc",
    }
    if since:
        params["since"] = since

    seen = written = skipped = pull_requests = 0
    for batch in github_api.paginate(url, params):
        for item in batch:
            # why: the issues endpoint returns PRs too, distinguished only by
            # this key. Number ranges overlap, so filtering here stops a PR from
            # overwriting the issue cache entry with the same number.
            if "pull_request" in item:
                pull_requests += 1
                continue
            seen += 1
            path = _cache_path(int(item["number"]))
            if path.exists():
                skipped += 1
            else:
                github_api.write_json(path, item)
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
