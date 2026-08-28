"""Settings object (phase 1). The load-bearing property: every default equals the
project's pre-existing behaviour, so an unconfigured clone is unchanged.
"""

import os

from factory_floor.config import COLLECTION_NAME, Settings, get_settings


class TestDefaults:
    def test_defaults_match_todays_literals(self):
        s = Settings()
        assert s.llm_model == "gpt-4.1-mini"
        assert s.llm_temperature == 0.0
        assert s.embedding_model == "text-embedding-3-small"
        assert s.collection_name == COLLECTION_NAME == "factory_floor_manuals"
        assert s.daily_spend_cap_usd is None
        assert s.cost_alert_threshold == 0.8
        assert s.safety_gate_mode == "rewrite"
        assert s.safety_gate_on_stream == "buffer"
        assert s.semantic_cache_enabled is False
        assert s.audit_enabled is True
        assert s.require_login is False
        assert s.tenant_id == "default"
        assert s.secrets_backend == "env"

    def test_frozen(self):
        s = Settings()
        try:
            s.llm_model = "x"
        except Exception as exc:  # FrozenInstanceError
            assert "cannot assign" in str(exc).lower() or "frozen" in type(exc).__name__.lower()
        else:
            raise AssertionError("Settings should be immutable")

    def test_from_env_with_nothing_set_equals_defaults(self, monkeypatch):
        for name in list(os.environ):
            if name.startswith("FACTORY_FLOOR_"):
                monkeypatch.delenv(name, raising=False)
        s = Settings.from_env()
        assert (s.llm_model, s.collection_name, s.safety_gate_mode) == (
            "gpt-4.1-mini",
            "factory_floor_manuals",
            "rewrite",
        )


class TestEnvOverrides:
    def test_string_override(self, monkeypatch):
        monkeypatch.setenv("FACTORY_FLOOR_LLM_MODEL", "gpt-5")
        assert Settings.from_env().llm_model == "gpt-5"

    def test_bool_parsing(self, monkeypatch):
        monkeypatch.setenv("FACTORY_FLOOR_SEMANTIC_CACHE_ENABLED", "true")
        assert Settings.from_env().semantic_cache_enabled is True
        monkeypatch.setenv("FACTORY_FLOOR_SEMANTIC_CACHE_ENABLED", "0")
        assert Settings.from_env().semantic_cache_enabled is False

    def test_optional_float_parsing(self, monkeypatch):
        assert Settings.from_env().daily_spend_cap_usd is None
        monkeypatch.setenv("FACTORY_FLOOR_DAILY_SPEND_CAP_USD", "2.50")
        assert Settings.from_env().daily_spend_cap_usd == 2.5

    def test_garbage_float_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("FACTORY_FLOOR_COST_ALERT_THRESHOLD", "not-a-number")
        assert Settings.from_env().cost_alert_threshold == 0.8


class TestGetSettingsCache:
    def test_cached_until_cleared(self, monkeypatch):
        first = get_settings()
        assert get_settings() is first  # same object -> cached
        monkeypatch.setenv("FACTORY_FLOOR_LLM_MODEL", "changed")
        assert get_settings() is first  # still cached, override not seen yet
        get_settings.cache_clear()
        assert get_settings().llm_model == "changed"
