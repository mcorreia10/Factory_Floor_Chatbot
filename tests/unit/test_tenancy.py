"""factory_floor/tenancy.py (phase 7) — the multi-tenant seam."""

from factory_floor.cache import SemanticCache
from factory_floor.tenancy import list_tenants, resolve_collection


class TestResolveCollection:
    def test_default_keeps_the_existing_collection(self):
        assert resolve_collection("default") == "factory_floor_manuals"
        assert resolve_collection(None) == "factory_floor_manuals"
        assert resolve_collection("") == "factory_floor_manuals"

    def test_other_tenant_gets_a_suffixed_collection(self):
        assert resolve_collection("acme") == "factory_floor_manuals__acme"

    def test_unsafe_characters_are_sanitised(self):
        assert resolve_collection("acme corp/eu") == "factory_floor_manuals__acme_corp_eu"

    def test_follows_a_configured_base_name(self, monkeypatch):
        monkeypatch.setenv("FACTORY_FLOOR_COLLECTION_NAME", "custom_base")
        from factory_floor.config import get_settings

        get_settings.cache_clear()
        assert resolve_collection("default") == "custom_base"
        assert resolve_collection("acme") == "custom_base__acme"


class TestListTenants:
    def test_reads_the_committed_registry(self):
        tenants = list_tenants()
        assert "default" in {t.tenant_id for t in tenants}
        default = next(t for t in tenants if t.tenant_id == "default")
        assert default.vector_collection == "factory_floor_manuals"


class TestCacheKeyIsTenantScoped:
    def test_entry_id_differs_by_tenant(self):
        a = SemanticCache._entry_id("q", "GENERAL", "VFD", "English", "acme")
        b = SemanticCache._entry_id("q", "GENERAL", "VFD", "English", "beta")
        assert a != b
