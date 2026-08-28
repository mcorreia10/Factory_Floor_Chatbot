"""Pure-logic tests for factory_floor/fault_codes.py — no vectorstore, no network.

Locks the 2026-08-25 exact-fault-code-lookup work (CLAUDE.md): the character-confusion
normalisation, the "a typo is not an unknown code" rule, and the definition-vs-cross-
reference scoring.
"""

from factory_floor.fault_codes import (
    extract_codes,
    extract_possible_codes,
    normalize_code,
    score_occurrence,
    suggest_similar,
)


class TestExtractCodes:
    def test_finds_codes_in_order_without_duplicates(self):
        text = "Drive shows F30021, then f30011, and F30021 again later."
        assert extract_codes(text) == ["F30021", "F30011"]

    def test_empty_and_codeless_text(self):
        assert extract_codes("") == []
        assert extract_codes(None) == []
        assert extract_codes("the motor is overheating and vibrating") == []

    def test_rejects_wrong_length_digit_runs(self):
        # 4 digits is too short, 6 is too long — the pattern is exactly [FA] + 5 digits.
        assert extract_codes("F3002 and F300211 are not valid codes") == []

    def test_alarm_codes_too(self):
        assert extract_codes("alarm A30015 is active") == ["A30015"]


class TestNormalizeCode:
    def test_letter_for_digit_confusions_in_numeric_part_only(self):
        assert normalize_code("F3OO21") == "F30021"
        assert normalize_code("f3oo21") == "F30021"
        assert normalize_code("A3II5S") == "A31155"

    def test_leading_letter_is_left_alone(self):
        # The O in a leading position is a real F/A only in theory here; the point is the
        # first character is never translated.
        assert normalize_code("F0000O").startswith("F")


class TestScoreOccurrence:
    def test_prefers_the_definition_over_a_cross_reference(self):
        text = (
            "See also: F30021 Note: this cross-reference is only relevant for chassis power units. "
            "Much later in the document: F30021 Power unit: Ground fault. Cause: earth fault."
        )
        score, context = score_occurrence(text, "F30021")
        assert score > 0
        assert context.startswith("Power unit")

    def test_missing_code_returns_sentinel_low_score(self):
        score, context = score_occurrence("nothing to see here", "F30021")
        assert score == -99
        assert context == ""


class TestExtractPossibleCodes:
    def test_typed_confusion_that_maps_onto_a_real_code(self):
        # F30021 is a real Siemens code in the committed fault_codes.csv.
        assert extract_possible_codes("display reads F3OO21 after a trip") == [("F3OO21", "F30021")]

    def test_well_formed_code_is_not_a_typo(self):
        assert extract_possible_codes("display reads F30021") == []

    def test_ordinary_words_are_not_codes(self):
        # "FOSSIL" matches the loose letter-or-digit shape but has no real digits.
        assert extract_possible_codes("this is a FOSSIL of a machine") == []

    def test_confusion_that_maps_onto_nothing_real(self):
        # Normalises to F99999, which is not in the corpus -> no suggestion invented.
        assert extract_possible_codes("saw F9999S on the panel") == []

    def test_empty(self):
        assert extract_possible_codes("") == []


class TestSuggestSimilar:
    def test_character_confusion_wins_outright(self):
        assert suggest_similar("F3OO21") == ["F30021"]
