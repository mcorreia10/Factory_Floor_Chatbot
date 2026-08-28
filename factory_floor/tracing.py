import os
from uuid import uuid4

from langsmith import utils as ls_utils

DEFAULT_PROJECT = "factory-floor"

BASE_TAGS = ["factory_floor"]


def configure_tracing(project_name: str = None) -> str:
    """Routes this project's traces to their own LangSmith project without touching
    .env, which is shared across unrelated projects and pins
    LANGSMITH_PROJECT=lca-lc-foundation (owner asked for it to be left alone).

    The assignment is unconditional on purpose: app.py imports this package before
    calling load_dotenv(), while notebooks call load_dotenv() before importing it.
    load_dotenv() defaults to override=False, so an unconditional os.environ set here
    wins under both import orders — os.environ.setdefault() would not (it would lose
    in the notebook case and leak traces into the shared project).

    The cache_clear() calls are not defensive noise: langsmith 0.11.0's get_env_var()
    and get_tracer_project() are functools.lru_cache'd, so a plain os.environ change
    made after the first traced call is silently ignored without this.
    """
    # Lazy import: this module is loaded from factory_floor/__init__.py before anything
    # else, so keep the top level dependency-free.
    from factory_floor.config import get_settings

    project = (
        project_name
        or os.environ.get("FACTORY_FLOOR_LANGSMITH_PROJECT")
        or get_settings().langsmith_project
        or DEFAULT_PROJECT
    )
    os.environ["LANGSMITH_PROJECT"] = project
    ls_utils.get_env_var.cache_clear()
    ls_utils.get_tracer_project.cache_clear()
    return project


def is_tracing_enabled() -> bool:
    return bool(ls_utils.tracing_is_enabled())


def tracing_endpoint() -> str:
    return os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


def trace_config(run_name: str, tags: list = None, metadata: dict = None, run_id=None) -> dict:
    """Builds a RunnableConfig-shaped dict for .invoke(..., config=...). run_id must be
    a fresh uuid.UUID per call — never reuse one across two invocations, and never pass
    a str (CallbackManager.on_chain_start expects an actual UUID object)."""
    return {
        "run_name": run_name,
        "tags": BASE_TAGS + list(tags or []),
        "metadata": {"component": run_name, **(metadata or {})},
        "run_id": run_id or uuid4(),
    }


def run_url(run_id, project_name: str = None) -> str:
    """Builds a LangSmith permalink for a run_id already returned by run_diagnostic_agent()
    or ask() — works uniformly for both, unlike the two mechanisms LangChain itself
    exposes: `LangChainTracer.get_run_url()` only sees runs created through the
    callback-manager path (agent.invoke(config=...)), and raises "No traced run found"
    for @traceable-decorated plain-Python calls like rag.ask() uses. This helper
    sidesteps both by calling Client._construct_run_url() directly with a minimal
    object exposing only `.id` — it needs no read access to the run itself (no
    `read_run()` call, which is both deprecated and requires project_id/start_time on
    SmithDB backends anyway), just the run_id and the project name/id to resolve the
    session. One cheap network call (read_project, to resolve the project name to an
    id) rather than zero, but no dependency on the run already being queryable.

    `get_run_url`/`_construct_run_url` are themselves deprecated (removal after
    2027-01-31, successor: `client.runs.get_url`) but functional as of langsmith 0.11.0
    — acceptable for this project's timeline.
    """
    from types import SimpleNamespace

    from langsmith import Client

    client = Client()
    project = project_name or os.environ.get("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    return client.get_run_url(run=SimpleNamespace(id=run_id), project_name=project)
