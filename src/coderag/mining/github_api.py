"""The only place in the pipeline that talks HTTP to GitHub.

Rate-limit handling, retries and cursor pagination live here so that every miner
behaves identically. Callers get parsed JSON and never see a Response.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests
import typer

from coderag.config import settings

API_ROOT = "https://api.github.com"
PER_PAGE = 100
MAX_ATTEMPTS = 5
# why: a margin, not 0. The remaining counter is per-token and another process
# (or a previous run's in-flight requests) can spend the last unit under us.
LOW_REMAINING = 2


def headers() -> dict[str, str]:
    """Standard GitHub REST headers, with the token when one is configured."""
    result = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        result["Authorization"] = f"Bearer {settings.github_token}"
    return result


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
    return max(0.0, float(reset) - datetime.now(timezone.utc).timestamp())


def _sleep_if_exhausted(response: requests.Response) -> None:
    """Block until the quota resets if this response used up the budget."""
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is None or int(remaining) > LOW_REMAINING:
        return
    wait = _seconds_until_reset(response) + 5
    typer.echo(f"  rate limit low ({remaining} left); sleeping {wait:.0f}s")
    time.sleep(wait)


def get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    """GET with retries that sleep on rate limits instead of hammering.

    404 is returned rather than raised: callers use it to decide whether a
    number is a pull request, and the empty answer is worth caching.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, headers=headers(), params=params, timeout=30)
        except requests.RequestException as error:
            # why: a mine is thousands of requests over hours, so a dropped
            # connection or read timeout is a certainty, not an edge case.
            # Without this the whole run dies on one reset packet.
            typer.echo(f"  {type(error).__name__}; retry {attempt}")
            time.sleep(2**attempt)
            continue
        if response.status_code in (200, 404):
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


def paginate(url: str, params: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
    """Yield successive pages, following GitHub's cursor.

    why: the Link header's rel="next", not an incrementing ?page=. GitHub
    refuses offset pagination past 10,000 items with a 422 and tells you to use
    its cursor; that URL already carries the cursor plus every filter applied.
    """
    while True:
        response = get(url, params)
        batch = response.json()
        if not batch:
            return
        yield batch
        next_url = response.links.get("next", {}).get("url")
        if not next_url:
            return
        url, params = next_url, {}


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically, so a kill cannot leave a truncated cache entry."""
    tmp = path.with_suffix(".json.tmp")
    # why: write-then-rename. A partial write would be indistinguishable from a
    # complete one on the next run, and the corruption would be permanent.
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def cached_object(path: Path, url: str) -> dict[str, Any]:
    """One JSON object, from disk if present, else fetched and cached."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = get(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return payload


def cached_list(path: Path, url: str) -> list[dict[str, Any]]:
    """A fully paginated list, from disk if present, else fetched and cached."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for batch in paginate(url, {"per_page": PER_PAGE}):
        items.extend(batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, items)
    return items
