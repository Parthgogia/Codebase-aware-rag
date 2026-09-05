"""Smoke test: the config actually loads and the paths resolve."""

from coderag.config import settings
from coderag.paths import CLONE_DIR, RESULTS_DIR


def test_settings_load_from_yaml() -> None:
    assert settings.repo_owner == "pandas-dev"
    assert len(settings.pinned_sha) == 40
    assert settings.eval_k_values == [1, 5, 10, 20]


def test_paths_are_absolute() -> None:
    assert CLONE_DIR.is_absolute()
    assert RESULTS_DIR.is_absolute()
