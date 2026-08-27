"""Paths and configuration for The Factory Floor.

The module-level ``Path`` constants and ``COLLECTION_NAME`` are unchanged — notebooks
and the app import them directly. The ``Settings`` object added below is the typed home
for everything the professionalization work needs to configure (model, spend cap, cache,
audit, tenant); it is read via ``get_settings()`` and layered on top, not instead of,
the constants.

Every knob has a safe default equal to today's behaviour and can be overridden with a
``FACTORY_FLOOR_*`` environment variable. ``.env`` itself is deliberately left untouched
(it is a shared personal file — see CLAUDE.md); document new knobs in ``.env.example``.
"""

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_DIR = PROJECT_ROOT / "data" / "manuals"
VECTOR_DIR = PROJECT_ROOT / "data" / "vectorstore"
SOURCES_CSV = PROJECT_ROOT / "manual_sources.csv"
MACHINES_CSV = PROJECT_ROOT / "machines.csv"
HISTORY_CSV = PROJECT_ROOT / "maintenance_history.csv"
COLLECTION_NAME = "factory_floor_manuals"

DEFECT_IMAGE_DIR = PROJECT_ROOT / "data" / "defect_images"
VISION_MODEL_DIR = PROJECT_ROOT / "data" / "vision_model"
DEFECT_MANIFEST_CSV = PROJECT_ROOT / "defect_image_manifest.csv"

EVAL_SCENARIOS_CSV = PROJECT_ROOT / "eval_scenarios.csv"
FAULT_CODES_CSV = PROJECT_ROOT / "fault_codes.csv"

OPERATORS_CSV = PROJECT_ROOT / "operators.csv"
TENANTS_CSV = PROJECT_ROOT / "tenants.csv"


# --- typed settings -----------------------------------------------------------------

def _env_str(name, default):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_opt_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """One immutable snapshot of configuration. Build it with ``Settings.from_env()``
    (what ``get_settings()`` does) or directly in a test."""

    # Credentials / models
    openai_api_key: str | None = None
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0
    embedding_model: str = "text-embedding-3-small"

    # Retrieval / tracing
    collection_name: str = COLLECTION_NAME
    langsmith_project: str | None = None

    # Cost control (phase 3) — off unless a cap is set
    daily_spend_cap_usd: float | None = None
    cost_alert_threshold: float = 0.8
    cost_ledger_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cost_ledger.jsonl")

    # Safety gate (phase 4)
    safety_gate_mode: str = "rewrite"          # off | rewrite | block
    safety_gate_on_stream: str = "buffer"      # buffer | disable_stream

    # Semantic cache (phase 6) — opt-in, same as rerank/code_aware
    semantic_cache_enabled: bool = False
    semantic_cache_similarity_threshold: float = 0.95
    semantic_cache_ttl_hours: int = 720

    # Audit trail + identity (phase 5)
    audit_enabled: bool = True
    require_login: bool = False
    audit_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "audit.sqlite3")

    # Multi-tenant (phase 7) — single tenant until a deployment needs otherwise
    tenant_id: str = "default"

    # Secrets source (phase 1) — "env" reads os.environ; others are design-only seams
    secrets_backend: str = "env"

    @classmethod
    def from_env(cls):
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            llm_model=_env_str("FACTORY_FLOOR_LLM_MODEL", "gpt-4.1-mini"),
            llm_temperature=_env_float("FACTORY_FLOOR_LLM_TEMPERATURE", 0.0),
            embedding_model=_env_str("FACTORY_FLOOR_EMBEDDING_MODEL", "text-embedding-3-small"),
            collection_name=_env_str("FACTORY_FLOOR_COLLECTION_NAME", COLLECTION_NAME),
            langsmith_project=_env_str("FACTORY_FLOOR_LANGSMITH_PROJECT", None),
            daily_spend_cap_usd=_env_opt_float("FACTORY_FLOOR_DAILY_SPEND_CAP_USD", None),
            cost_alert_threshold=_env_float("FACTORY_FLOOR_COST_ALERT_THRESHOLD", 0.8),
            cost_ledger_path=Path(
                _env_str("FACTORY_FLOOR_COST_LEDGER_PATH", str(PROJECT_ROOT / "data" / "cost_ledger.jsonl"))
            ),
            safety_gate_mode=_env_str("FACTORY_FLOOR_SAFETY_GATE_MODE", "rewrite"),
            safety_gate_on_stream=_env_str("FACTORY_FLOOR_SAFETY_GATE_ON_STREAM", "buffer"),
            semantic_cache_enabled=_env_bool("FACTORY_FLOOR_SEMANTIC_CACHE_ENABLED", False),
            semantic_cache_similarity_threshold=_env_float(
                "FACTORY_FLOOR_SEMANTIC_CACHE_SIMILARITY_THRESHOLD", 0.95
            ),
            semantic_cache_ttl_hours=_env_int("FACTORY_FLOOR_SEMANTIC_CACHE_TTL_HOURS", 720),
            audit_enabled=_env_bool("FACTORY_FLOOR_AUDIT_ENABLED", True),
            require_login=_env_bool("FACTORY_FLOOR_REQUIRE_LOGIN", False),
            audit_db_path=Path(
                _env_str("FACTORY_FLOOR_AUDIT_DB_PATH", str(PROJECT_ROOT / "data" / "audit.sqlite3"))
            ),
            tenant_id=_env_str("FACTORY_FLOOR_TENANT_ID", "default"),
            secrets_backend=_env_str("FACTORY_FLOOR_SECRETS_BACKEND", "env"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings snapshot, read from the environment once.

    Cached, so anything that mutates the environment afterwards (notably ``app.py``
    calling ``load_dotenv()`` *after* importing this package) must call
    ``get_settings.cache_clear()`` before the first real use. Tests do this via an
    autouse fixture.
    """
    return Settings.from_env()


# Introspection helper for .env.example / docs generation and tests.
SETTINGS_FIELD_NAMES = tuple(f.name for f in fields(Settings))
