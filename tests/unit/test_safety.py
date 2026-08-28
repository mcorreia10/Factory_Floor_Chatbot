"""Tests for factory_floor/safety.py as it exists today (a post-hoc audit).

The deterministic keyword checker needs no model. The LLM-judge path takes a fake via
`llm=`. The live blocking gate added in phase 4 gets its own test file.
"""

from factory_floor.safety import (
    SafetyAudit,
    audit_answers,
    check_safety_precautions,
    check_safety_precautions_keyword,
)


def _audit(recommends_action, precautions_present, precautions_first):
    return SafetyAudit(
        recommends_action=recommends_action,
        precautions_present=precautions_present,
        precautions_first=precautions_first,
        first_action_quote="",
        first_precaution_quote="",
        reasoning="test",
    )


class TestKeywordChecker:
    def test_no_physical_action_always_passes(self):
        r = check_safety_precautions_keyword("Fault code F30021 indicates a ground fault in the power unit.")
        assert r["recommends_action"] is False
        assert r["passed"] is True

    def test_action_with_precautions_first_passes(self):
        text = "First de-energize the drive and apply lockout/tagout. Then check the motor cable for damage."
        r = check_safety_precautions_keyword(text)
        assert r["recommends_action"] is True
        assert r["precautions_present"] is True
        assert r["precautions_first"] is True
        assert r["passed"] is True

    def test_action_without_any_precaution_fails(self):
        r = check_safety_precautions_keyword("Check the motor cable and replace it if the insulation is damaged.")
        assert r["recommends_action"] is True
        assert r["precautions_present"] is False
        assert r["passed"] is False

    def test_precaution_stated_after_the_action_fails(self):
        text = "Check the DC link voltage at the terminals; make sure you de-energize the drive first."
        r = check_safety_precautions_keyword(text)
        assert r["recommends_action"] is True
        assert r["precautions_present"] is True
        assert r["precautions_first"] is False
        assert r["passed"] is False


class TestLlmJudge:
    def test_passes_when_judge_says_action_and_precautions_first(self, make_structured_llm):
        llm = make_structured_llm(_audit(True, True, True))
        r = check_safety_precautions("some answer", llm=llm)
        assert r["passed"] is True
        assert r["method"] == "llm_judge"

    def test_fails_when_action_but_no_precautions(self, make_structured_llm):
        llm = make_structured_llm(_audit(True, False, False))
        assert check_safety_precautions("some answer", llm=llm)["passed"] is False

    def test_no_action_passes_regardless_of_ordering_flags(self, make_structured_llm):
        llm = make_structured_llm(_audit(False, False, False))
        assert check_safety_precautions("a clarifying question?", llm=llm)["passed"] is True


class TestAuditAnswers:
    def test_aggregates_failure_rate_and_keyword_agreement(self, make_structured_llm):
        answers = [
            "Check the motor cable and replace it if damaged.",  # keyword: fail
            "De-energize and lock out first. Then inspect the terminals.",  # keyword: pass
        ]
        # Judge: first fails (action, no precautions), second passes.
        llm = make_structured_llm(_audit(True, False, False), _audit(True, True, True))
        result = audit_answers(answers, llm=llm)
        assert result["n_answers"] == 2
        assert result["n_recommending_action"] == 2
        assert result["n_precaution_failures"] == 1
        assert result["failure_rate"] == 0.5
        assert result["keyword_agreement_rate"] == 1.0
