import textwrap
import json
import json_repair
import logging
from datetime import date
import ollama
from infrastructure.config import OLLAMA_MODEL, OLLAMA_TEMP, OLLAMA_MAX_TOKENS, CONFLICT_LOW_THRESHOLD, CONFLICT_MEDIUM_THRESHOLD, REQUIRED_KEYS, REQUIRED_SIGNAL_KEYS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback default output
DEFAULT_OUTPUT = {
    "stance": "neutral",
    "confidence": 0.0,
    "conflict_level": "medium",
    "signals": {
        "news": "unavailable",
        "chart": "unavailable",
        "timeseries": "unavailable",
        "positioning": "unavailable"
    },
    "reasoning": "LLM fusion failed after maximum retries. Defaulting to neutral stance."
}

def compute_conflict(modality_outputs: dict) -> tuple[str, str, float]:
    """Computes conflict level using confidence-weighted signal imbalance/agreement.

    - Compares total bullish vs bearish confidence across modalities
    - Replaces naive majority voting (counting signals in feature prototype) with confidence-weighted aggregation
    - Sums bullish and bearish confidence separately across modalities (neutral signals ignored)
    - Computes imbalance = abs(bull - bear) / (bull + bear)
    - Higher imbalance = stronger agreement (lower conflict),  neutral signals are ignored

    Returns tuple of (conflict_level, conflict_note, imbalance).
    imbalance is the raw float (0.0-1.0) so callers (eg. the dashboard) can display/bucket it directly instead of parsing it back out of conflict_note's string.
    """
    # Sum confidence where models say its bullish
    weighted_bull = sum(
        m["confidence"] for m in modality_outputs.values()
        if m["signal"] == 1
    )
    # Sum confidence where models say its bearish
    weighted_bear = sum(
        m["confidence"] for m in modality_outputs.values()
        if m["signal"] == -1
    )
    total = weighted_bull + weighted_bear # combined total
    # If no directional signal (0), means neutral
    # * imbalance is reported as 0.0 here. not a "measured" agreement, just nothing to disagree on
    if total == 0:
        return "low", "all modalities neutral", 0.0

    # Calculate imbalance (how one-sided signals are) using basic normalized difference
    imbalance = abs(weighted_bull - weighted_bear) / total

    # Higher imbalance -> stronger agreement -> lower conflict
    # Lower imbalance -> greater disagreement -> higher conflict
    if imbalance > CONFLICT_LOW_THRESHOLD:
        conflict_level = "low"
        conflict_note = f"strong directional agreement (imbalance={imbalance:.2f})"
    elif imbalance > CONFLICT_MEDIUM_THRESHOLD:
        conflict_level = "medium"
        conflict_note = f"partial agreement (imbalance={imbalance:.2f})"
    else:
        conflict_level = "high"
        conflict_note = f"significant disagreement (imbalance={imbalance:.2f})"

    logger.info(f"Conflict: {conflict_level} | weighted_bull={weighted_bull:.3f} weighted_bear={weighted_bear:.3f} imbalance={imbalance:.3f}")
    return conflict_level, conflict_note, imbalance

def build_fusion_prompt(pair: str, timeframe: str, modality_outputs: dict, conflict_level: str, conflict_note: str) -> str:
    """Builds structured prompt for the fusion LLM.

    - Passes signal + confidence (already weight-adjusted if feedback loop multiplier applied)
    - Include conflict signal to help LLM with reasoning behaviour
    - LLM is responsible for stance + final reasoning

    Returns a very structured multi-line output string.
    """
    # All the modalities + current date
    news = modality_outputs["news"]
    chart = modality_outputs["chart"]
    ts = modality_outputs["timeseries"]
    pos = modality_outputs["positioning"]
    today = date.today().isoformat()

    # Nested helper to format confidence display
    def conf_line(data: dict) -> str:
        # Show weight-adjusted confidence if weighting multiplier applied (!= 0)
        # * else default to 1.0 if no feedback history
        w = data.get("weight", 1.0)
        
        # Raw confidence from original model output (before any weigting)
        raw = data.get("raw_confidence", data.get("confidence", 0.0))

        # Effective confidence AFTER weighting, what Fusion layer will actually use
        eff = data.get("confidence", 0.0)

        # If weight ~1.0 means no adjustment applied, just show final confidence
        if abs(w - 1.0) < 0.001:
            return f"confidence: {eff}"
        
        # But if there is weightings applied, show all the components for transparency
        return (
            f"raw_confidence: {raw} | weight: {w} | "
            f"effective_confidence: {eff} "
            f"(weight reflects historical reliability for this pair)"
        )
    
    # Imporved prompt from feature prototype 
    # * Initially it was hedging response and acknowledging 'no conflicts' which is noise in reasoning
    #   old feature prototype prompt - "<2-3 sentences synthesising overall stance, acknowledging any conflicts>"
    # * In iteration 2, we increase the prompt to "4-5" instead of "2-3" sentence as UAT suggest it was too brief
    prompt = f"""
            You are a systematic forex market analyst for {pair} on the {timeframe} timeframe.
            Analysis date: {today}
            You will receive structured outputs from four independent analysis models.
            Synthesise them into a final trading decision.

            Return ONLY a valid JSON object. No explanation, no markdown, no code blocks. Raw JSON only.

            MODALITY OUTPUTS:

            1. News/Sentiment (ProsusAI/finbert)
            signal: {news['signal']} (+1=bullish, -1=bearish, 0=neutral)
            {conf_line(news)}
            distribution: {news.get('distribution', {})}

            2. Chart Pattern (foduucom YOLO)
            signal: {chart['signal']} (+1=bullish, -1=bearish, 0=neutral)
            {conf_line(chart)}
            detected_pattern: {chart.get('pattern', 'none')}

            3. Time-Series (Custom GRU)
            signal: {ts['signal']} (+1=bullish, -1=bearish, 0=neutral)
            {conf_line(ts)}
            direction_probability: {ts.get('direction_probability', 0.5)}

            4. Retail Positioning (contrarian rule-based)
            signal: {pos['signal']} (+1=bullish, -1=bearish, 0=neutral)
            {conf_line(pos)}
            retail_long: {pos.get('long_pct', 'N/A')}%
            retail_short: {pos.get('short_pct', 'N/A')}%

            Pre-computed conflict level: {conflict_level} ({conflict_note})

            REQUIRED OUTPUT FORMAT:

            {{
            "stance": "bullish" or "bearish" or "neutral",
            "confidence": <float 0.0 to 1.0>,
            "conflict_level": "low" or "medium" or "high",
            "signals": {{
                "news": "<one sentence based on finbert output above>",
                "chart": "<one sentence based on YOLO output above>",
                "timeseries": "<one sentence based on GRU output above>",
                "positioning": "<one sentence based on positioning data above>"
            }},
            "reasoning": "<4-5 sentences synthesising overall stance. Explain what drove the stance, not just what modality said. If conflict_level is low, state the conclusion directly without hedging. Only acknowledge conflicts if conflict_level is medium or high. If any modality has a weight below 1.0, you may note it carries reduced influence for this pair.>"
            }}
            """
    
    # Use text wrap to cut away the whitespaces in the prompt
    # * i added those whitespaces so me and viewers can read the code better
    return textwrap.dedent(prompt).strip()

def extract_json(raw_text: str) -> dict:
    """Extracts and parses JSON from LLM response.

    Use json_repaire library instead of manual brace matching + regex like we did in feature prototype,
    its more clean and handles more cases:
    - Strips markdown code fences ```, in case LLM mess up even after specifying output
    - Ignores braces {} within quoted strings
    - Auto-fix missing braces (close unclosed brackets/braces) if JSON gets truncated
    
    Returns parsed JSON dict or ValueError if no valid JSON.
    """
    result = json_repair.loads(raw_text)

    # json_repair returns {} instead of raising
    # * so we guard with ValueError if truly no result
    if not result:
        raise ValueError("No JSON object found in LLM response.")

    return result

def validate_output(parsed: dict) -> list[str]:
    """Validates parsed LLM output against required schema.

    - Top-level keys required: stance, confidence, conflict_level, signals, reasoning
    - Signals must include: news, chart, timeseries, positioning
    - stance must be either {bullish, bearish, neutral}
    - confidence between [0.0, 1.0]
    - conflict_level either {low, medium, high}

    Returns list of strings with problems, if valid return empty string.
    """
    issues = []
    # Check for required top level keys
    missing_top = REQUIRED_KEYS - set(parsed.keys())
    if missing_top:
        issues.append(f"missing top-level keys: {missing_top}")

    # Check modality keys inside signals
    missing_signals = REQUIRED_SIGNAL_KEYS - set(parsed.get("signals", {}).keys())
    if missing_signals:
        issues.append(f"missing signal keys: {missing_signals}")

    # Validate the stance
    stance = parsed.get("stance")
    if stance not in ["bullish", "bearish", "neutral"]:
        issues.append(f"invalid stance value: {parsed.get('stance')}")

    # Confidence validation
    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)):
        issues.append("confidence is not a number")
    elif not (0.0 <= conf <= 1.0):
        issues.append(f"confidence out of range: {conf} (expected 0.0~1.0)")

    # Validate conflict level
    conflict = parsed.get("conflict_level")
    if conflict not in ["low", "medium", "high"]:
        issues.append(f"invalid conflict_level value: {parsed.get('conflict_level')}")

    return issues

def call_llm(prompt: str, retries: int = 3) -> dict:
    """Calls LLM with fusion prompt and enforces valid structured output.
    
    - Retries up to 3 times on malformed JSON or validation failure

    Returns validated output dict or fallback version (DEFAULT_OUTPUT)
    """
    for attempt in range(1, retries + 1):
        logger.info(f"LLM call attempt {attempt}/{retries}...")
        try:
            # Use Ollama LLM (mistral and its configs from config.py)
            # * we do like in feature prototype so we can change any model in config if decide to eg. Llama3
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": OLLAMA_TEMP,
                    "num_predict": OLLAMA_MAX_TOKENS
                }
            )
            raw = response["message"]["content"]
            logger.debug(f"Raw LLM response: {raw}")

            # Extract structured JSON from model output
            parsed = extract_json(raw)
            issues = validate_output(parsed) # validate schema if correct before accept

            if issues:
                logger.warning(f"Attempt {attempt} validation issues: {issues}")
                continue # retry

            logger.info("LLM output valid.")
            return parsed

        # JSON extract or parse fail
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt} JSON decode error: {e}")
        # API fail or runtime error (may forsee this get triggered if too much traffic)
        except Exception as e:
            logger.error(f"Attempt {attempt} LLM call failed: {e}")

    logger.error("All LLM retry attempts exhausted, returning default output.")
    return DEFAULT_OUTPUT

def get_fusion_output(pair: str, timeframe: str, modality_outputs: dict) -> dict:
    """Main entry point for the fusion layer.

    - Validate required modality output exist first;
    - Computes conflict, builds prompt, calls LLM, returns structured validated output

    Returns full structured output dict (stance, confidence, conflict_level, signals, reasoning) from LLM.
    """
    required_modalities = {"news", "chart", "timeseries", "positioning"}
    missing = required_modalities - set(modality_outputs.keys())
    if missing:
        raise ValueError(f"Missing modality outputs: {missing}")

    conflict_level, conflict_note, imbalance = compute_conflict(modality_outputs)
    prompt = build_fusion_prompt(pair, timeframe, modality_outputs, conflict_level, conflict_note)
    result = call_llm(prompt)

    # Attach the raw imbalance value (computed here, not by the LLM) onto the returned dict
    # * so the dashboard can display/bucket it directly without re-deriving or string-parsing conflict_note
    result = {**result, "conflict_imbalance": imbalance}

    logger.info(f"Fusion output for {pair} {timeframe}: stance={result.get('stance')} confidence={result.get('confidence')} conflict={result.get('conflict_level')}")
    return result


# Run smoke test with python -m core.fusion
if __name__ == "__main__":
    # Using inputs only
    test_inputs = {
        "news": {"signal": 1, "confidence": 0.72, "distribution": {"positive": 3, "negative": 2, "neutral": 2}},
        "chart": {"signal": -1, "confidence": 0.35, "pattern": "Head and shoulders top"},
        "timeseries": {"signal": -1, "confidence": 0.444, "direction_probability": 0.444},
        "positioning": {"signal": 1, "confidence": 0.62, "long_pct": 38, "short_pct": 62}
    }
    result = get_fusion_output("EURUSD", "1h", test_inputs)
    print(json.dumps(result, indent=2))