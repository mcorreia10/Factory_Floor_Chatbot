"""factory_floor.secrets.get_secret (phase 1) — the vault seam."""

import pytest

from factory_floor.secrets import get_secret


class TestEnvBackend:
    def test_reads_from_environment(self, monkeypatch):
        monkeypatch.setenv("FF_TEST_SECRET", "s3cr3t")
        assert get_secret("FF_TEST_SECRET") == "s3cr3t"

    def test_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("FF_TEST_SECRET", raising=False)
        assert get_secret("FF_TEST_SECRET") is None
        assert get_secret("FF_TEST_SECRET", "fallback") == "fallback"


class TestOtherBackends:
    @pytest.mark.parametrize("backend", ["aws", "vault", "doppler", "sops"])
    def test_design_only_backends_raise_not_implemented(self, monkeypatch, backend):
        monkeypatch.setenv("FACTORY_FLOOR_SECRETS_BACKEND", backend)
        with pytest.raises(NotImplementedError):
            get_secret("OPENAI_API_KEY")

    def test_unknown_backend_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("FACTORY_FLOOR_SECRETS_BACKEND", "banana")
        with pytest.raises(ValueError):
            get_secret("OPENAI_API_KEY")
