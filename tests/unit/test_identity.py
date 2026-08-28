"""factory_floor/identity.py (phase 5) — operator sign-in against operators.csv."""

from factory_floor.identity import Operator, authenticate, hash_pin, list_operators, new_salt


class TestHashPin:
    def test_deterministic_for_same_pin_and_salt(self):
        salt = "abc123"
        assert hash_pin("1234", salt) == hash_pin("1234", salt)

    def test_different_salt_changes_the_hash(self):
        assert hash_pin("1234", "saltA") != hash_pin("1234", "saltB")

    def test_new_salt_is_random(self):
        assert new_salt() != new_salt()


class TestListOperators:
    def test_reads_the_committed_roster(self):
        ops = list_operators()
        assert all(isinstance(o, Operator) for o in ops)
        assert "OP-1001" in {o.operator_id for o in ops}


class TestAuthenticate:
    def test_correct_id_and_pin(self):
        op = authenticate("OP-1001", "1234")
        assert op is not None
        assert op.name == "Ana Costa"
        assert op.role == "technician"
        assert op.tenant_id == "default"

    def test_wrong_pin_returns_none(self):
        assert authenticate("OP-1001", "0000") is None

    def test_unknown_operator_returns_none(self):
        assert authenticate("OP-9999", "1234") is None

    def test_empty_pin_returns_none(self):
        assert authenticate("OP-1001", "") is None
