"""factory_floor/safety.py::enforce_safety — the live blocking gate (phase 4).

The judge and the rewrite both run on the injected fake model (make_gate_llm), so no
network. The pre-existing post-hoc audit functions are covered in test_safety.py.
"""

from langchain_core.messages import AIMessage

from factory_floor.safety import FIXED_HELD_FALLBACK, SafetyAudit, enforce_safety


def _audit(recommends_action=True, precautions_present=False, precautions_first=False):
    return SafetyAudit(
        recommends_action=recommends_action,
        precautions_present=precautions_present,
        precautions_first=precautions_first,
        first_action_quote="",
        first_precaution_quote="",
        reasoning="test",
    )


ACTION_NO_PRECAUTION = "Check the motor cable [SOURCE 1] and replace it if the insulation is damaged."
CLARIFYING_QUESTION = "Is the drive still powered, or has it already been isolated?"


class TestCheapPasses:
    def test_mode_off_always_passes_unchanged(self, make_gate_llm):
        gate = enforce_safety(ACTION_NO_PRECAUTION, llm=make_gate_llm([]), mode="off")
        assert gate.action == "pass"
        assert gate.delivered_answer == ACTION_NO_PRECAUTION

    def test_no_physical_action_passes_without_calling_the_judge(self):
        # No llm passed; if the judge were called it would try to build a real one and _no_network raises.
        gate = enforce_safety(CLARIFYING_QUESTION, mode="rewrite")
        assert gate.action == "pass"
        assert "judge" not in gate.audit

    def test_empty_answer_passes(self):
        assert enforce_safety("", mode="rewrite").action == "pass"


class TestJudgeSaysOk:
    def test_action_with_precautions_first_passes(self, make_gate_llm):
        llm = make_gate_llm([_audit(recommends_action=True, precautions_present=True, precautions_first=True)])
        gate = enforce_safety("Isolate and lock out. [SOURCE 1] Then measure the DC link.", llm=llm, mode="rewrite")
        assert gate.action == "pass"
        assert gate.audit["judge"]["passed"] is True


class TestRewrite:
    def test_rewrites_when_the_rewrite_passes_and_keeps_citations(self, make_gate_llm):
        # judge: original fails, re-check of the rewrite passes
        llm = make_gate_llm(
            [_audit(True, False, False), _audit(True, True, True)],
            rewrite_text="Safety precautions: de-energize, lock out, verify no voltage. "
                         "[SOURCE 1] Then check the motor cable and replace it if damaged.",
        )
        gate = enforce_safety(ACTION_NO_PRECAUTION, llm=llm, mode="rewrite")
        assert gate.action == "rewritten"
        assert "Safety precautions" in gate.delivered_answer
        assert "[SOURCE 1]" in gate.delivered_answer
        assert gate.audit["citations_preserved"] is True

    def test_holds_when_the_rewrite_still_fails_the_recheck(self, make_gate_llm):
        llm = make_gate_llm(
            [_audit(True, False, False), _audit(True, False, False)],  # re-check still fails
            rewrite_text="Check the motor cable [SOURCE 1] first, then think about de-energizing maybe.",
        )
        gate = enforce_safety(ACTION_NO_PRECAUTION, llm=llm, mode="rewrite")
        assert gate.action == "held"
        assert gate.delivered_answer == FIXED_HELD_FALLBACK

    def test_holds_when_the_rewrite_drops_a_citation(self, make_gate_llm):
        llm = make_gate_llm(
            [_audit(True, False, False), _audit(True, True, True)],  # re-check would pass...
            rewrite_text="Safety precautions: de-energize and lock out. Then check the cable.",  # ...but no [SOURCE 1]
        )
        gate = enforce_safety(ACTION_NO_PRECAUTION, llm=llm, mode="rewrite")
        assert gate.action == "held"
        assert gate.audit["citations_preserved"] is False

    def test_holds_when_the_rewrite_drops_a_fault_code(self, make_gate_llm):
        original = "For F30021, check the motor cable [SOURCE 1] for a short to earth."
        llm = make_gate_llm(
            [_audit(True, False, False), _audit(True, True, True)],
            rewrite_text="Safety precautions: de-energize, lock out. [SOURCE 1] Then check the motor cable to earth.",
        )
        gate = enforce_safety(original, llm=llm, mode="rewrite")
        assert gate.action == "held"  # F30021 missing from the rewrite


class TestBlockMode:
    def test_block_mode_never_rewrites(self, make_gate_llm):
        llm = make_gate_llm([_audit(True, False, False)])
        gate = enforce_safety(ACTION_NO_PRECAUTION, llm=llm, mode="block")
        assert gate.action == "held"
        assert gate.delivered_answer == FIXED_HELD_FALLBACK
        assert "recheck" not in gate.audit  # no rewrite attempted


class TestCostConfigForwarding:
    """The gate's judge/rewrite are real LLM spend. Until the run config was forwarded
    they were invisible to the session cost line and the daily cap (and sat outside the
    agent's trace tree). These assert the config actually reaches both call sites."""

    class _RecordingLLM:
        def __init__(self, audits, rewrite_text="Safety precautions: isolate first. [SOURCE 1] Then check."):
            self._audits = list(audits)
            self._rewrite_text = rewrite_text
            self.configs = []

        def with_structured_output(self, schema):
            outer = self

            class _Runnable:
                def invoke(self, messages, config=None, **kwargs):
                    outer.configs.append(config)
                    return outer._audits.pop(0)

            return _Runnable()

        def invoke(self, messages, config=None, **kwargs):
            self.configs.append(config)
            return AIMessage(content=self._rewrite_text)

    def test_config_reaches_the_judge(self):
        llm = self._RecordingLLM([_audit(precautions_present=True, precautions_first=True)])
        cfg = {"callbacks": ["sentinel"]}
        enforce_safety(ACTION_NO_PRECAUTION, llm=llm, mode="rewrite", config=cfg)
        assert llm.configs == [cfg]

    def test_config_reaches_rewrite_and_recheck(self):
        llm = self._RecordingLLM([
            _audit(),                                                    # original fails
            _audit(precautions_present=True, precautions_first=True),    # rewrite passes
        ])
        cfg = {"callbacks": ["sentinel"]}
        gate = enforce_safety(ACTION_NO_PRECAUTION, llm=llm, mode="rewrite", config=cfg)
        assert gate.action == "rewritten"
        assert llm.configs == [cfg, cfg, cfg]  # judge, rewrite, re-check

    def test_no_config_is_still_valid(self):
        llm = self._RecordingLLM([_audit(precautions_present=True, precautions_first=True)])
        enforce_safety(ACTION_NO_PRECAUTION, llm=llm, mode="rewrite")
        assert llm.configs == [None]


PT_ACTION_NO_PRECAUTION = (
    "Verifique o cabo do motor [SOURCE 1] e substitua-o se o isolamento estiver danificado."
)


class TestNonEnglishStillReachesTheJudge:
    """The keyword shortcut is English-only. Before this, a Portuguese answer matched no
    action verb, scored recommends_action=False and took the cheap pass — the safety gate
    silently did nothing for 4 of the 5 languages the app offers."""

    def test_portuguese_action_answer_is_judged_not_waved_through(self, make_gate_llm):
        # The judge says it fails; if the shortcut were still trusted we would never get
        # here and the answer would come back "pass" unchanged.
        gate = enforce_safety(
            PT_ACTION_NO_PRECAUTION,
            llm=make_gate_llm([_audit(), _audit()]),  # original fails, rewrite fails too
            mode="block",
            language="Portuguese",
        )
        assert gate.action == "held"

    def test_english_keeps_the_cheap_path(self, make_gate_llm):
        # No action verb, English: still passes without consulting the judge. Passing an
        # empty audit list proves the judge was never called (it would raise otherwise).
        gate = enforce_safety(CLARIFYING_QUESTION, llm=make_gate_llm([]), language="English")
        assert gate.action == "pass"
        assert gate.reason == "no physical action instructed"

    def test_keyword_reliability_is_recorded(self, make_gate_llm):
        gate = enforce_safety(
            PT_ACTION_NO_PRECAUTION,
            llm=make_gate_llm([_audit(precautions_present=True, precautions_first=True)]),
            language="Portuguese",
        )
        assert gate.audit["keyword_reliable"] is False
