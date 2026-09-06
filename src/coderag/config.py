"""Typed configuration for the whole pipeline.

Precedence, highest first: environment variables (CODERAG_* and GITHUB_TOKEN),
then .env, then config.yaml, then the defaults declared below.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# why: derived from this file's location rather than os.getcwd(), so `pytest`,
# `python -m coderag.x` and an editor all resolve the same root no matter where
# they were launched from. parents[2] == <root>/src/coderag/config.py -> <root>.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Every tunable parameter in one place. Read config.yaml, not this file."""

    model_config = SettingsConfigDict(
        yaml_file=PROJECT_ROOT / "config.yaml",
        env_file=PROJECT_ROOT / ".env",
        env_prefix="CODERAG_",
        extra="forbid",
    )

    # --- target repository -------------------------------------------------
    repo_owner: str = "pandas-dev"
    repo_name: str = "pandas"
    # why: the index is only meaningful relative to one exact tree state, so the
    # SHA lives in config and is read by every stage. Passing it as a CLI flag
    # would let two stages silently disagree about which commit they indexed.
    pinned_sha: str

    # --- directories (relative entries are resolved in paths.py) -----------
    clone_dir: Path = Path("data/raw/repo")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    index_dir: Path = Path("data/index")
    results_dir: Path = Path("results")

    # --- secrets -----------------------------------------------------------
    # why: alias, not the CODERAG_ prefix. GITHUB_TOKEN is the name every other
    # tool (gh, CI) already uses, and there is no reason to make users set it twice.
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")

    # --- retrieval knobs ---------------------------------------------------
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 32

    eval_k_values: list[int] = [1, 5, 10, 20]

    # --- file inventory ----------------------------------------------------
    # why: these live in config rather than as constants in repo.py because the
    # inventory defines which files can ever be retrieved, which is an
    # experimental parameter, not an implementation detail.
    inventory_exclude: list[str] = [".git", "build", "doc/_build", "__pycache__"]
    vendored_markers: list[str] = ["_vendor", "vendored", "third_party"]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert config.yaml below env vars so env always wins."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


# why: one module-level instance. The config is read-only after start-up, and a
# get_settings() factory would only invite different stages to build different
# Settings objects from a mutated environment mid-run.
settings = Settings()
