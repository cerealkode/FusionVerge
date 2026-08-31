"""
Unit tests for the fusion modality (fusion.py)
Tests conflict scoring, prompt building, JSON extraction, schema validation, retry logic, and the full get_fusion_output entrypoint.
Ollama-dependent tests are skipped gracefully if Ollama is not reachable.

Usage - python -m pytest tests/test_fusion.py -v --disable-warnings
"""

import json
import pytest
import ollama
from unittest.mock import patch
from core.fusion import compute_conflict, build_fusion_prompt, extract_json, validate_output, call_llm, get_fusion_output, DEFAULT_OUTPUT
from infrastructure.config import CONFLICT_LOW_THRESHOLD, CONFLICT_MEDIUM_THRESHOLD


# Ollama availability check
def ollama_reachable() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False

requires_ollama = pytest.mark.skipif(
    not ollama_reachable(),
    reason="Ollama not reachable. Start Ollama and run 'ollama pull mistral' first."
)


# Shared test fixtures
@pytest.fixture
def all_bullish():
    return {
        "news": {"signal": 1, "confidence": 0.80},
        "chart": {"signal": 1, "confidence": 0.65},
        "timeseries": {"signal": 1, "confidence": 0.72},
        "positioning": {"signal": 1, "confidence": 0.60},
    }

@pytest.fixture
def all_bearish():
    return {
        "news": {"signal": -1, "confidence": 0.80},
        "chart": {"signal": -1, "confidence": 0.65},
        "timeseries": {"signal": -1, "confidence": 0.72},
        "positioning": {"signal": -1, "confidence": 0.60},
    }

@pytest.fixture
def two_v_two():
    return {
        "news": {"signal": 1, "confidence": 0.70},
        "chart": {"signal": 1, "confidence": 0.70},
        "timeseries": {"signal": -1, "confidence": 0.70},
        "positioning": {"signal": -1, "confidence": 0.70},
    }

@pytest.fixture
def all_neutral():
    return {
        "news": {"signal": 0, "confidence": 0.50},
        "chart": {"signal": 0, "confidence": 0.00},
        "timeseries": {"signal": 0, "confidence": 0.50},
        "positioning": {"signal": 0, "confidence": 0.50},
    }

@pytest.fixture
def mixed_with_neutral():
    """Three bullish, one neutral - neutral excluded from imbalance calc."""
    return {
        "news": {"signal": 1, "confidence": 0.80},
        "chart": {"signal": 1, "confidence": 0.75},
        "timeseries": {"signal": 1, "confidence": 0.70},
        "positioning": {"signal": 0, "confidence": 0.50},
    }

@pytest.fixture
def valid_modality_outputs():
    """Full modality output dict with all optional fields for prompt building."""
    return {
        "news": {
            "signal": 1, "confidence": 0.72,
            "distribution": {"positive": 3, "negative": 1, "neutral": 2}
        },
        "chart": {
            "signal": -1, "confidence": 0.35,
            "pattern": "Head and shoulders top"
        },
        "timeseries": {
            "signal": 1, "confidence": 0.61,
            "direction_probability": 0.61
        },
        "positioning": {
            "signal": 1, "confidence": 0.62,
            "long_pct": 38, "short_pct": 62
        },
    }

@pytest.fixture
def valid_llm_response():
    return {
        "stance": "bullish",
        "confidence": 0.68,
        "conflict_level": "medium",
        "signals": {
            "news": "Positive sentiment driven by ECB rate expectations.",
            "chart": "Head and shoulders top detected suggesting potential reversal.",
            "timeseries": "GRU model predicts upward movement with moderate confidence.",
            "positioning": "Retail traders heavily short, contrarian signal bullish.",
        },
        # 4-5 sentence are expected
        "reasoning": (
            "Three of four modalities align bullish despite a bearish chart signal. "
            "Conflict is medium given one dissenting modality. "
            "Overall stance is cautiously bullish. "
            "News sentiment carried the most weight given its higher confidence relative to the conflicting chart signal."
        ),
    }


# compute_conflict() function testing
class TestComputeConflict:
    def test_all_bullish_is_low_conflict(self, all_bullish):
        """Four agreeing bullish signals -> imbalance=1.0 -> low conflict."""
        level, _, _ = compute_conflict(all_bullish)
        assert level == "low"

    def test_all_bearish_is_low_conflict(self, all_bearish):
        """Four agreeing bearish signals -> imbalance=1.0 -> low conflict."""
        level, _, _ = compute_conflict(all_bearish)
        assert level == "low"

    def test_two_v_two_is_high_conflict(self, two_v_two):
        """Equal-weight 2v2 split -> imbalance=0.0 -> high conflict."""
        level, _, _ = compute_conflict(two_v_two)
        assert level == "high"

    def test_all_neutral_is_low_conflict(self, all_neutral):
        """All neutral -> total=0 -> low conflict (no disagreement)."""
        level, _, _ = compute_conflict(all_neutral)
        assert level == "low"

    def test_neutral_excluded_from_imbalance(self, mixed_with_neutral):
        """Neutral signal should not dilute the imbalance of the directional signals."""
        level, _, _ = compute_conflict(mixed_with_neutral)
        # Three bullish and zero bearish -> imbalance=1.0 -> low conflict
        assert level == "low"

    def test_confidence_weighting_changes_result(self):
        """Same signal directions but very different confidences
        - the high-confidence side should dominate and push toward low conflict even when numeric 2v2 split
        """
        skewed = {
            "news": {"signal": 1, "confidence": 0.95},
            "chart": {"signal": 1, "confidence": 0.90},
            "timeseries": {"signal": -1, "confidence": 0.10},
            "positioning": {"signal": -1, "confidence": 0.10},
        }
        level, _, _ = compute_conflict(skewed)
        # weighted_bull=1.85, weighted_bear=0.20, imbalance=0.80 -> low conflict
        assert level == "low"

    def test_medium_conflict_range(self):
        """Imbalance between CONFLICT_MEDIUM_THRESHOLD and CONFLICT_LOW_THRESHOLD
        should produce medium conflict.

        - test only meaningful if constructed imbalance actually falls in the medium band
        - so we assert the precondition directly against the config values rather than magic numbers
          eg. "assert CONFLICT_MEDIUM_THRESHOLD < imbalance < CONFLICT_LOW_THRESHOLD" instead of "assert 0.25 < imbalance < 0.50"
        """
        # * weighted_bull = 0.40 + 0.40 = 0.80
        # * weighted_bear = 0.40
        # * imbalance = |0.80 - 0.40| / 1.20 = 0.333-ish
        moderate = {
            "news": {"signal": 1, "confidence": 0.40},
            "chart": {"signal": 1, "confidence": 0.40},
            "timeseries": {"signal": -1, "confidence": 0.40},
            "positioning": {"signal": 0, "confidence": 0.50},
        }
        imbalance = round(0.40 / 1.20, 10)
        assert CONFLICT_MEDIUM_THRESHOLD < imbalance < CONFLICT_LOW_THRESHOLD, (
            f"Test precondition failed: imbalance {imbalance} is not in the medium band "
            f"({CONFLICT_MEDIUM_THRESHOLD}, {CONFLICT_LOW_THRESHOLD}). "
            f"Update the fixture or check config thresholds."
        )
        level, _, _ = compute_conflict(moderate)
        assert level == "medium"

    def test_returns_tuple(self, all_bullish):
        result = compute_conflict(all_bullish)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_conflict_level_is_valid_string(self, all_bullish):
        level, _, _ = compute_conflict(all_bullish)
        assert level in {"low", "medium", "high"}

    def test_missing_modality_raises(self):
        incomplete = {
            "news": {"signal": 1, "confidence": 0.7},
            "chart": {"signal": 1, "confidence": 0.6},
        }
        # compute_conflict() itself doesnt validate keys as thats get_fusion_output() job
        # * so should not crash if less than 4 modals
        level, _, _ = compute_conflict(incomplete)
        assert level in {"low", "medium", "high"}


# build_fusion_prompt() function testing
class TestBuildFusionPrompt:
    def test_contains_pair(self, valid_modality_outputs):
        prompt = build_fusion_prompt(
            "EURUSD", "1h", valid_modality_outputs, "medium", "partial agreement"
        )
        assert "EURUSD" in prompt

    def test_contains_timeframe(self, valid_modality_outputs):
        prompt = build_fusion_prompt(
            "EURUSD", "1h", valid_modality_outputs, "medium", "partial agreement"
        )
        assert "1h" in prompt

    def test_contains_all_modality_signals(self, valid_modality_outputs):
        prompt = build_fusion_prompt(
            "EURUSD", "1h", valid_modality_outputs, "medium", "partial agreement"
        )
        # All four signal values must appear in the prompt
        assert "signal: 1" in prompt # news and positioning
        assert "signal: -1" in prompt # chart

    def test_contains_conflict_level(self, valid_modality_outputs):
        prompt = build_fusion_prompt(
            "EURUSD", "1h", valid_modality_outputs, "high", "significant disagreement"
        )
        assert "high" in prompt
        assert "significant disagreement" in prompt

    def test_contains_required_output_format(self, valid_modality_outputs):
        """Prompt must instruct the LLM to return the required JSON schema."""
        prompt = build_fusion_prompt(
            "EURUSD", "1h", valid_modality_outputs, "low", "strong agreement"
        )
        assert "stance" in prompt
        assert "confidence" in prompt
        assert "conflict_level" in prompt
        assert "reasoning" in prompt

    def test_within_context_window(self, valid_modality_outputs):
        """Mistral 7B context window is 32k tokens.
        
        - rough upper bound is arnd 4 char per token, so 32k * 4 = 128k chars
        - prompt should not exceed this (note that this is just smart estimating)
        """
        prompt = build_fusion_prompt(
            "EURUSD", "1h", valid_modality_outputs, "medium", "partial agreement"
        )
        assert len(prompt) < 128_000

    def test_returns_string(self, valid_modality_outputs):
        prompt = build_fusion_prompt(
            "EURUSD", "1h", valid_modality_outputs, "low", "agreement"
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_different_pairs_produce_different_prompts(self, valid_modality_outputs):
        p1 = build_fusion_prompt("EURUSD", "1h", valid_modality_outputs, "low", "x")
        p2 = build_fusion_prompt("GBPUSD", "1h", valid_modality_outputs, "low", "x")
        assert p1 != p2


# extract_json() function testin
class TestExtractJson:
    def test_clean_json_parsed(self, valid_llm_response):
        raw = json.dumps(valid_llm_response)
        result = extract_json(raw)
        assert result["stance"] == "bullish"

    def test_strips_markdown_fences(self, valid_llm_response):
        raw = "```json\n" + json.dumps(valid_llm_response) + "\n```"
        result = extract_json(raw)
        assert result["stance"] == "bullish"

    def test_strips_plain_code_fence(self, valid_llm_response):
        raw = "```\n" + json.dumps(valid_llm_response) + "\n```"
        result = extract_json(raw)
        assert result["stance"] == "bullish"

    def test_json_with_leading_text(self, valid_llm_response):
        """LLM sometimes prepends explanation text before the JSON."""
        raw = "Here is my analysis:\n" + json.dumps(valid_llm_response)
        result = extract_json(raw)
        assert result["stance"] == "bullish"

    def test_auto_closes_missing_brace(self):
        """Long response with LLM truncation should still parse.
        
        - the fallback brace-closing logic should take care of this scenario
        - that logic is made for edge case, and this is just to test that logic itself
        """
        truncated = '{"stance": "bullish", "signals": {"news": "positive"}'
        result = extract_json(truncated)
        assert result["stance"] == "bullish"
        assert result["signals"]["news"] == "positive"

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("This response has no JSON at all.")

    def test_empty_string_raises(self):
        with pytest.raises((ValueError, Exception)):
            extract_json("")


# validate_output() function testing
class TestValidateOutput:
    def test_valid_output_returns_no_issues(self, valid_llm_response):
        issues = validate_output(valid_llm_response)
        assert issues == []

    def test_missing_stance_flagged(self, valid_llm_response):
        bad = {k: v for k, v in valid_llm_response.items() if k != "stance"}
        issues = validate_output(bad)
        assert any("stance" in i for i in issues)

    def test_missing_confidence_flagged(self, valid_llm_response):
        bad = {k: v for k, v in valid_llm_response.items() if k != "confidence"}
        issues = validate_output(bad)
        assert any("confidence" in i for i in issues)

    def test_missing_conflict_level_flagged(self, valid_llm_response):
        bad = {k: v for k, v in valid_llm_response.items() if k != "conflict_level"}
        issues = validate_output(bad)
        assert any("conflict_level" in i for i in issues)

    def test_missing_reasoning_flagged(self, valid_llm_response):
        bad = {k: v for k, v in valid_llm_response.items() if k != "reasoning"}
        issues = validate_output(bad)
        assert any("reasoning" in i for i in issues)

    def test_invalid_stance_value_flagged(self, valid_llm_response):
        bad = {**valid_llm_response, "stance": "hold"}
        issues = validate_output(bad)
        assert any("stance" in i for i in issues)

    def test_invalid_conflict_level_flagged(self, valid_llm_response):
        bad = {**valid_llm_response, "conflict_level": "none"}
        issues = validate_output(bad)
        assert any("conflict_level" in i for i in issues)

    def test_confidence_as_string_flagged(self, valid_llm_response):
        bad = {**valid_llm_response, "confidence": "0.7"}
        issues = validate_output(bad)
        assert any("confidence" in i for i in issues)

    def test_confidence_above_one_flagged(self, valid_llm_response):
        """LLM hallucinating a confidence > 1.0 should be caught and trigger a retry."""
        bad = {**valid_llm_response, "confidence": 1.5}
        issues = validate_output(bad)
        assert any("confidence" in i for i in issues)

    def test_confidence_below_zero_flagged(self, valid_llm_response):
        """Negative confidence should be caught and trigger a retry."""
        bad = {**valid_llm_response, "confidence": -0.1}
        issues = validate_output(bad)
        assert any("confidence" in i for i in issues)

    def test_confidence_exactly_zero_valid(self, valid_llm_response):
        """0.0 is a valid confidence (eg. DEFAULT_OUTPUT uses it)."""
        r = {**valid_llm_response, "confidence": 0.0}
        issues = validate_output(r)
        assert not any("confidence" in i for i in issues)

    def test_confidence_exactly_one_valid(self, valid_llm_response):
        """1.0 is a valid confidence boundary."""
        r = {**valid_llm_response, "confidence": 1.0}
        issues = validate_output(r)
        assert not any("confidence" in i for i in issues)

    def test_missing_signal_subkey_flagged(self, valid_llm_response):
        bad = dict(valid_llm_response)
        bad["signals"] = {k: v for k, v in valid_llm_response["signals"].items()
                          if k != "chart"}
        issues = validate_output(bad)
        assert any("chart" in i for i in issues)

    def test_all_valid_stances_accepted(self, valid_llm_response):
        for stance in ["bullish", "bearish", "neutral"]:
            r = {**valid_llm_response, "stance": stance}
            issues = validate_output(r)
            assert not any("stance" in i for i in issues)

    def test_all_valid_conflict_levels_accepted(self, valid_llm_response):
        for level in ["low", "medium", "high"]:
            r = {**valid_llm_response, "conflict_level": level}
            issues = validate_output(r)
            assert not any("conflict_level" in i for i in issues)


# call_llm() function testing
# * on the RETRY logic
# * mocked so no Ollama dependency
class TestCallLlmRetry:
    def _make_response(self, content: str):
        return {"message": {"content": content}}

    def test_valid_response_returned_on_first_attempt(self, valid_llm_response):
        raw = json.dumps(valid_llm_response)
        with patch("core.fusion.ollama.chat", return_value=self._make_response(raw)):
            result = call_llm("test prompt")
        assert result["stance"] == "bullish"

    def test_retries_on_malformed_json(self, valid_llm_response):
        """First two responses are garbage, third is valid - should succeed on attempt 3."""
        good = json.dumps(valid_llm_response)
        responses = [
            self._make_response("not json at all"),
            self._make_response("also not json"),
            self._make_response(good),
        ]
        with patch("core.fusion.ollama.chat", side_effect=responses):
            result = call_llm("test prompt", retries=3)
        assert result["stance"] == "bullish"

    def test_retries_on_validation_failure(self, valid_llm_response):
        """First response fails validation (bad stance), second is valid."""
        bad = json.dumps({**valid_llm_response, "stance": "hold"})
        good = json.dumps(valid_llm_response)
        responses = [
            self._make_response(bad),
            self._make_response(good),
        ]
        with patch("core.fusion.ollama.chat", side_effect=responses):
            result = call_llm("test prompt", retries=2)
        assert result["stance"] == "bullish"

    def test_returns_default_after_all_retries_exhausted(self):
        """All retries return malformed JSON -> should fall back to DEFAULT_OUTPUT."""
        with patch("core.fusion.ollama.chat", return_value=self._make_response("garbage")):
            result = call_llm("test prompt", retries=3)
        assert result == DEFAULT_OUTPUT

    def test_returns_default_on_ollama_exception(self):
        """If Ollama raises an exception on every attempt -> DEFAULT_OUTPUT."""
        with patch("core.fusion.ollama.chat", side_effect=Exception("connection refused")):
            result = call_llm("test prompt", retries=3)
        assert result == DEFAULT_OUTPUT

    def test_default_output_has_required_keys(self):
        """DEFAULT_OUTPUT must itself pass structural checks so pipeline never crashes."""
        for key in ["stance", "confidence", "conflict_level", "signals", "reasoning"]:
            assert key in DEFAULT_OUTPUT
        assert DEFAULT_OUTPUT["stance"] == "neutral"


# get_fusion_output() component/entrypoint tests
# * end-to-end within the fusion module (LLM mocked)
# * with 1 INTEGRATION test - test_live_llm_call()
class TestGetFusionOutput:
    def test_returns_required_keys(self, valid_modality_outputs, valid_llm_response):
        with patch("core.fusion.ollama.chat",
                   return_value={"message": {"content": json.dumps(valid_llm_response)}}):
            result = get_fusion_output("EURUSD", "1h", valid_modality_outputs)
        for key in ["stance", "confidence", "conflict_level", "signals", "reasoning"]:
            assert key in result

    def test_missing_modality_raises(self, valid_modality_outputs):
        incomplete = {k: v for k, v in valid_modality_outputs.items() if k != "chart"}
        with pytest.raises(ValueError, match="Missing modality outputs"):
            get_fusion_output("EURUSD", "1h", incomplete)

    def test_stance_is_valid(self, valid_modality_outputs, valid_llm_response):
        with patch("core.fusion.ollama.chat",
                   return_value={"message": {"content": json.dumps(valid_llm_response)}}):
            result = get_fusion_output("EURUSD", "1h", valid_modality_outputs)
        assert result["stance"] in {"bullish", "bearish", "neutral"}

    def test_confidence_in_range(self, valid_modality_outputs, valid_llm_response):
        with patch("core.fusion.ollama.chat",
                   return_value={"message": {"content": json.dumps(valid_llm_response)}}):
            result = get_fusion_output("EURUSD", "1h", valid_modality_outputs)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_neutral_modalities_completes_without_error(self, valid_llm_response):
        """
        All-neutral modality outputs should still produce a valid fusion result.

        - exercises the total==0 branch in compute_conflict
        - confirms the fusion pipeline completes without crashing
        """
        all_neutral = {
            "news": {"signal": 0, "confidence": 0.50, "distribution": {}},
            "chart": {"signal": 0, "confidence": 0.00, "pattern": "none"},
            "timeseries": {"signal": 0, "confidence": 0.50, "direction_probability": 0.5},
            "positioning": {"signal": 0, "confidence": 0.50, "long_pct": 50, "short_pct": 50},
        }
        with patch("core.fusion.ollama.chat",
                   return_value={"message": {"content": json.dumps(valid_llm_response)}}):
            result = get_fusion_output("EURUSD", "1h", all_neutral)
        assert result["stance"] in {"bullish", "bearish", "neutral"}
        assert result["conflict_level"] in {"low", "medium", "high"}

    @requires_ollama
    def test_live_llm_call(self, valid_modality_outputs):
        """Full real Ollama call - only runs if Ollama is reachable."""
        result = get_fusion_output("EURUSD", "1h", valid_modality_outputs)
        assert result["stance"] in {"bullish", "bearish", "neutral"}
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["conflict_level"] in {"low", "medium", "high"}