"""factory_floor/recurrence.py — the zero-cost prior-occurrence check.

No model is involved anywhere in this module, so these are pure-logic tests: given rows,
what does it report? The history lookup is monkeypatched so the real CSV and SQLite are
never touched.
"""

import pytest

from factory_floor import recurrence


def _event(date, code, action, event_type="fault", outcome="", who="A. Silva", downtime="1.0"):
    return {
        "event_date": date,
        "fault_code": code,
        "action_taken": action,
        "event_type": event_type,
        "outcome": outcome,
        "technician": who,
        "downtime_hours": downtime,
    }


@pytest.fixture
def fake_history(monkeypatch):
    def _install(rows):
        monkeypatch.setattr(recurrence, "get_machine_history", lambda mid, **kw: list(rows))
    return _install


class TestCodeInQuestion:
    def test_finds_a_single_code(self):
        assert recurrence.code_in_question("F30805 keeps coming back") == "F30805"

    def test_none_when_no_code(self):
        assert recurrence.code_in_question("the motor is making a noise") is None

    def test_none_when_two_different_codes(self):
        # "Has THIS fault happened before?" has no single answer with two codes in play.
        assert recurrence.code_in_question("I see F30805 and also F30021") is None


class TestFindPriorOccurrences:
    def test_matches_only_the_requested_code_and_sorts_oldest_first(self, fake_history):
        fake_history([
            _event("2024-08-15", "F30805", "Replaced the power unit module"),
            _event("2021-01-05", "F07860", "Checked external interlock wiring"),
            _event("2020-10-30", "F30805", "Replaced the power unit module"),
        ])
        found = recurrence.find_prior_occurrences("VFD-04", "F30805")
        assert [r["event_date"] for r in found] == ["2020-10-30", "2024-08-15"]

    def test_general_and_blank_machine_return_nothing(self, fake_history):
        fake_history([_event("2020-10-30", "F30805", "x")])
        assert recurrence.find_prior_occurrences("GENERAL", "F30805") == []
        assert recurrence.find_prior_occurrences("", "F30805") == []


class TestSummarize:
    def test_empty_is_safe(self):
        s = recurrence.summarize([])
        assert s["count"] == 0 and s["recurring"] is False

    def test_counts_gaps_and_flags_recurrence(self, fake_history):
        fake_history([
            _event("2020-10-30", "F30805", "Replaced the power unit module"),
            _event("2024-08-15", "F30805", "Replaced the power unit module"),
        ])
        s = recurrence.summarize(recurrence.find_prior_occurrences("VFD-04", "F30805"))
        assert s["count"] == 2
        assert s["recurring"] is True
        assert s["shortest_gap_days"] == 1385
        # Years apart: it returned, but not "the fix did not hold" fast.
        assert s["returned_quickly"] is False

    def test_returned_quickly_when_inside_the_window(self, fake_history):
        fake_history([
            _event("2026-01-01", "F30805", "Power cycled the drive"),
            _event("2026-01-08", "F30805", "Power cycled the drive"),
        ])
        s = recurrence.summarize(recurrence.find_prior_occurrences("VFD-04", "F30805"))
        assert s["shortest_gap_days"] == 7
        assert s["returned_quickly"] is True

    def test_tallies_outcomes_and_marks_operator_notes(self, fake_history):
        fake_history([
            _event("2026-01-01", "F30805", "Replaced module", outcome="resolved"),
            _event("2026-02-01", "F30805", "Power cycle", event_type="operator_resolution",
                   outcome="temporary"),
        ])
        s = recurrence.summarize(recurrence.find_prior_occurrences("VFD-04", "F30805"))
        assert s["outcomes"] == {"resolved": 1, "temporary": 1}
        assert [a["source"] for a in s["actions"]] == ["logbook", "operator"]

    def test_never_recommends_an_action(self, fake_history):
        # The module reports; it must not grow a "suggested action" key. Ranking by
        # frequency would recommend the power cycle 2-to-1 here, which is the wrong
        # advice for a fault that keeps coming back.
        fake_history([
            _event("2026-01-01", "F30805", "Replaced the power unit module"),
            _event("2026-01-05", "F30805", "Power cycle"),
            _event("2026-01-09", "F30805", "Power cycle"),
        ])
        s = recurrence.summarize(recurrence.find_prior_occurrences("VFD-04", "F30805"))
        assert "recommendation" not in s and "suggested_action" not in s


class TestPriorOccurrenceReport:
    def test_none_without_a_code(self, fake_history):
        fake_history([_event("2020-10-30", "F30805", "x")])
        assert recurrence.prior_occurrence_report("VFD-04", "the drive is noisy") is None

    def test_none_when_the_code_is_new_to_this_machine(self, fake_history):
        fake_history([_event("2020-10-30", "F30805", "x")])
        assert recurrence.prior_occurrence_report("VFD-04", "F30021 ground fault") is None

    def test_reports_when_seen_before(self, fake_history):
        fake_history([
            _event("2020-10-30", "F30805", "Replaced the power unit module"),
            _event("2024-08-15", "F30805", "Replaced the power unit module"),
        ])
        report = recurrence.prior_occurrence_report("VFD-04", "F30805 again this morning")
        assert report["fault_code"] == "F30805"
        assert report["machine_id"] == "VFD-04"
        assert report["count"] == 2
