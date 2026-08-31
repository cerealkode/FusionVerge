"""
Evaluation for orchestrator (core/orchestrator.py)

- To mainly answer hypothesis question 1:
    CONSISTENCY - runs the full pipeline on EURUSD 1h 10 consecutive times.
    BASELINE COMPARISON - compares the full pipeline against three baselines (price-only, text-only, and naive LLM).
- Requires Ollama running with Mistral, trained models in models/, and live API access (RSS feeds, MyFxBook, market data)

Usage - python evaluation/eval_orchestrator.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
import time
import statistics
import logging
import ollama
from core.orchestrator import Orchestrator
from modalities.sentiment import fetch_headlines, get_sentiment_signal, load_sentiment_model
from modalities.chart import fetch_ohlcv
from modalities.positioning import get_positioning_signal
from modalities.timeseries import get_timeseries_signal
from infrastructure.config import OLLAMA_MODEL, OLLAMA_TEMP, OLLAMA_MAX_TOKENS

# Suppress INFO-level logging noise from core/httpx/ollama - this script's
# print() calls are the only output that should reach the console.
logging.getLogger().setLevel(logging.WARNING)
for _name in ("core", "modalities", "httpx", "ollama", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)


# SECTION 1 - Consistency
CONSISTENCY_PAIR = "EURUSD"
CONSISTENCY_TIMEFRAME = "1h"
CONSISTENCY_RUNS = 10

def run_consistency(orch: Orchestrator):
    print(f"\n{'='*75}")
    print(f"SECTION 1 - CONSISTENCY EVALUATION")
    print(f"Pair: {CONSISTENCY_PAIR} {CONSISTENCY_TIMEFRAME} | Runs: {CONSISTENCY_RUNS}")
    print(f"{'='*75}\n")

    results = []

    for i in range(1, CONSISTENCY_RUNS + 1):
        print(f"Run {i}/{CONSISTENCY_RUNS}...", end=" ", flush=True)
        start = time.time()
        try:
            output = orch.run(CONSISTENCY_PAIR, CONSISTENCY_TIMEFRAME)
            elapsed = round(time.time() - start, 1)
            results.append({
                "run": i,
                "stance": output["stance"],
                "confidence": output["confidence"],
                "conflict_level": output["conflict_level"],
                "elapsed": elapsed,
                "error": None,
            })
            print(f"stance={output['stance']:<10} conf={output['confidence']:.3f} "
                  f"conflict={output['conflict_level']:<8} ({elapsed}s)")
        except Exception as e:
            elapsed = round(time.time() - start, 1)
            print(f"ERROR: {e}")
            results.append({
                "run": i, "stance": "ERROR", "confidence": None,
                "conflict_level": None, "elapsed": elapsed, "error": str(e),
            })

    valid = [r for r in results if r["error"] is None]
    if not valid:
        print("\nAll runs failed. Check Ollama is running and models are available.")
        return

    # Stance distribution
    stances = [r["stance"] for r in valid]
    stance_counts = {}
    for s in stances:
        stance_counts[s] = stance_counts.get(s, 0) + 1
    most_common = max(stance_counts, key=stance_counts.get)
    consistency_rate = stance_counts[most_common] / len(stances) * 100

    # Confidence statistics
    confs = [r["confidence"] for r in valid]
    conf_mean = statistics.mean(confs)
    conf_stdev = statistics.stdev(confs) if len(confs) > 1 else 0.0

    # Conflict distribution
    conflict_counts = {}
    for r in valid:
        c = r["conflict_level"]
        conflict_counts[c] = conflict_counts.get(c, 0) + 1

    print(f"\n{'='*75}")
    print("RESULTS TABLE")
    print(f"{'='*75}")
    header = f"{'Run':<5} {'Stance':<12} {'Confidence':<12} {'Conflict':<10} {'Time (s)'}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["error"]:
            print(f"{r['run']:<5} ERROR: {r['error']}")
        else:
            print(f"{r['run']:<5} {r['stance']:<12} {r['confidence']:<12.3f} "
                  f"{r['conflict_level']:<10} {r['elapsed']}")

    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print(f"Successful runs   : {len(valid)} / {CONSISTENCY_RUNS}")
    print(f"\nStance distribution:")
    for stance, count in sorted(stance_counts.items(), key=lambda x: -x[1]):
        print(f"  {stance:<12} {count:>2}x  ({count/len(valid)*100:.0f}%)")
    print(f"\nConsistency rate  : {consistency_rate:.0f}% ({most_common} was most common)")
    print(f"\nConfidence stats:")
    print(f"  Mean   : {conf_mean:.3f}")
    print(f"  StdDev : {conf_stdev:.3f}")
    print(f"  Min    : {min(confs):.3f}")
    print(f"  Max    : {max(confs):.3f}")
    print(f"\nConflict distribution:")
    for level, count in sorted(conflict_counts.items(), key=lambda x: -x[1]):
        print(f"  {level:<10} {count:>2}x  ({count/len(valid)*100:.0f}%)")
    print(f"{'='*75}")


# -------------------------------------
# SECTION 2 - Baseline comparisons
BASELINE_PAIRS = [
    ("EURUSD", "1h"),
    ("GBPUSD", "1h"),
    ("USDJPY", "1h"),
    ("AUDUSD", "1d"),
    ("USDCAD", "1d"),
]


def build_baseline_prompt(pair: str, timeframe: str, raw_data: dict) -> str:
    """
    Naive baseline prompt - feeds raw data directly to Mistral with no
    specialist model preprocessing. No FinBERT scores, no YOLO patterns,
    no GRU probabilities. Represents what a non-specialist LLM approach
    produces on the same underlying data.
    """
    headlines = raw_data.get("headlines", [])
    ohlc = raw_data.get("ohlc", {})
    positioning = raw_data.get("positioning", {})
    headlines_str = (
        "\n".join(f"  - {h}" for h in headlines[:10])
        if headlines else "  (none available)"
    )

    return f"""You are a forex market analyst for {pair} on the {timeframe} timeframe.
Analyse the following raw market data and provide a trading stance.

Return ONLY a valid JSON object with no markdown or explanation.

=== RAW DATA ===

Recent headlines mentioning {pair}:
{headlines_str}

Latest OHLC (most recent candle):
  Open:  {ohlc.get('open', 'N/A')}
  High:  {ohlc.get('high', 'N/A')}
  Low:   {ohlc.get('low', 'N/A')}
  Close: {ohlc.get('close', 'N/A')}

Retail positioning (MyFxBook community):
  Long:  {positioning.get('long_pct', 'N/A')}%
  Short: {positioning.get('short_pct', 'N/A')}%

=== REQUIRED OUTPUT ===

{{
  "stance": "bullish" or "bearish" or "neutral",
  "confidence": <float 0.0 to 1.0>,
  "conflict_level": "low" or "medium" or "high",
  "signals": {{
    "news": "<one sentence summarising headline sentiment>",
    "chart": "<one sentence on price action from OHLC>",
    "timeseries": "<one sentence on recent price trend>",
    "positioning": "<one sentence on retail positioning>"
  }},
  "reasoning": "<2-3 sentences explaining your overall stance>"
}}"""


def fetch_raw_data(pair: str, timeframe: str) -> dict:
    """
    Fetches raw unenriched data for the baseline prompt - no specialist
    model processing applied. Headlines are unclassified, OHLC is the
    last candle only, positioning is the raw percentage numbers.
    """
    raw = {}
    try:
        raw["headlines"] = fetch_headlines(pair, max_headlines=10)
    except Exception:
        raw["headlines"] = []

    try:
        df = fetch_ohlcv(pair, timeframe)
        last = df.iloc[-1]
        raw["ohlc"] = {
            "open":  round(float(last["Open"]), 5),
            "high":  round(float(last["High"]), 5),
            "low":   round(float(last["Low"]), 5),
            "close": round(float(last["Close"]), 5),
        }
    except Exception:
        raw["ohlc"] = {}

    try:
        pos = get_positioning_signal(pair)
        raw["positioning"] = {
            "long_pct":  pos.get("long_pct"),
            "short_pct": pos.get("short_pct"),
        }
    except Exception:
        raw["positioning"] = {}

    return raw


def call_baseline_llm(prompt: str) -> dict:
    """Calls Mistral directly with raw data - no retry, no validation."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": OLLAMA_TEMP, "num_predict": OLLAMA_MAX_TOKENS}
        )
        raw = response["message"]["content"]
        text = re.sub(r"```json|```", "", raw).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return {"stance": "ERROR", "reasoning": f"No JSON found: {raw[:200]}"}
        return json.loads(text[start:end])
    except Exception as e:
        return {"stance": "ERROR", "reasoning": str(e)}


def get_baseline1_price_only(pair: str, timeframe: str) -> dict:
    """
    Baseline 1 (price-only, PPR §3.7.2): the time-series modality's signal
    taken alone, with no fusion and no other modality involved. Equivalent
    to a traditional technical-indicator-only approach. GRU always commits
    to a direction (no neutral output), so stance is always bullish/bearish.
    """
    try:
        result = get_timeseries_signal(pair, timeframe)
        stance = "bullish" if result["signal"] == 1 else "bearish"
        return {"stance": stance, "confidence": result["confidence"], "error": None}
    except Exception as e:
        return {"stance": "ERROR", "confidence": 0.0, "error": str(e)}


def get_baseline2_text_only(pair: str, pipe) -> dict:
    """
    Baseline 2 (text-only, PPR §3.7.2): the sentiment modality's signal
    taken alone, with no fusion and no other modality involved. Equivalent
    to a pure news-driven approach. Can return neutral where headlines are
    absent, ambiguous, or below the aggregation confidence threshold.
    """
    try:
        result = get_sentiment_signal(pair, pipe=pipe)
        if result["signal"] == 1:
            stance = "bullish"
        elif result["signal"] == -1:
            stance = "bearish"
        else:
            stance = "neutral"
        return {"stance": stance, "confidence": result["confidence"], "error": None}
    except Exception as e:
        return {"stance": "ERROR", "confidence": 0.0, "error": str(e)}


def run_baseline_comparison(orch: Orchestrator, sentiment_pipe):
    print(f"\n{'='*75}")
    print("SECTION 2 - BASELINE COMPARISON")
    print("Full pipeline vs Baseline 1 (price-only) vs Baseline 2 (text-only)")
    print("vs Baseline 3 (naive LLM), on the same underlying data.")
    print(f"{'='*75}\n")

    pipeline_results = []
    baseline1_results = []
    baseline2_results = []
    baseline_results = []

    for pair, timeframe in BASELINE_PAIRS:
        print(f"\n{'─'*75}")
        print(f"Analysing {pair} {timeframe}...")

        # Full pipeline
        print(f"  [Pipeline]  Running...")
        try:
            out = orch.run(pair, timeframe)
            pipeline_results.append({
                "pair": pair, "timeframe": timeframe,
                "stance": out["stance"],
                "confidence": out.get("confidence", 0),
                "conflict": out.get("conflict_level", "?"),
                "reasoning": out.get("reasoning", ""),
                "error": None,
            })
            print(f"  [Pipeline]  stance={out['stance']} conf={out.get('confidence', 0):.3f}")
        except Exception as e:
            print(f"  [Pipeline]  ERROR: {e}")
            pipeline_results.append({
                "pair": pair, "timeframe": timeframe,
                "stance": "ERROR", "confidence": 0, "conflict": "?",
                "reasoning": str(e), "error": str(e),
            })

        # Baseline 1 - price-only (timeseries modality alone)
        b1 = get_baseline1_price_only(pair, timeframe)
        baseline1_results.append({"pair": pair, "timeframe": timeframe, **b1})
        print(f"  [Baseline1] stance={b1['stance']} conf={b1['confidence']:.3f}  (price-only)")

        # Baseline 2 - text-only (sentiment modality alone)
        b2 = get_baseline2_text_only(pair, sentiment_pipe)
        baseline2_results.append({"pair": pair, "timeframe": timeframe, **b2})
        print(f"  [Baseline2] stance={b2['stance']} conf={b2['confidence']:.3f}  (text-only)")

        # Baseline 3 - naive LLM
        print(f"  [Baseline3] Fetching raw data and calling Mistral...")
        raw_data = fetch_raw_data(pair, timeframe)
        prompt = build_baseline_prompt(pair, timeframe, raw_data)
        b_out = call_baseline_llm(prompt)
        baseline_results.append({
            "pair": pair, "timeframe": timeframe,
            "stance": b_out.get("stance", "ERROR"),
            "confidence": b_out.get("confidence", 0),
            "conflict": b_out.get("conflict_level", "?"),
            "reasoning": b_out.get("reasoning", ""),
            "error": None if b_out.get("stance") != "ERROR" else b_out.get("reasoning"),
        })
        print(f"  [Baseline3] stance={b_out.get('stance', 'ERROR')} "
              f"conf={b_out.get('confidence', 0):.3f}  (naive LLM)")

    # Compact summary table (for screenshot / direct copy into report table)
    print(f"\n{'='*75}")
    print("SUMMARY TABLE")
    print(f"{'='*75}")
    header = (f"{'Pair':<12}{'Pipeline':<22}{'Baseline1 (price)':<22}"
              f"{'Baseline2 (text)':<22}{'Baseline3 (LLM)'}")
    print(header)
    print("-" * len(header))
    for p, b1, b2, b3 in zip(pipeline_results, baseline1_results, baseline2_results, baseline_results):
        p_str  = f"{p['stance']}, {p['confidence']:.3f}"
        b1_str = f"{b1['stance']}, {b1['confidence']:.3f}"
        b2_str = f"{b2['stance']}, {b2['confidence']:.3f}"
        b3_str = f"{b3['stance']}, {b3['confidence']:.3f}"
        label = p['pair'] + ' ' + p['timeframe']
        print(f"{label:<12}{p_str:<22}{b1_str:<22}{b2_str:<22}{b3_str}")

    # Directional agreement tally - pipeline stance vs each baseline's stance
    print(f"\n{'='*75}")
    print("DIRECTIONAL AGREEMENT WITH PIPELINE")
    print(f"{'='*75}")
    for name, results in [("Baseline 1 (price-only)", baseline1_results),
                           ("Baseline 2 (text-only)", baseline2_results),
                           ("Baseline 3 (naive LLM)", baseline_results)]:
        agree = sum(
            1 for p, b in zip(pipeline_results, results)
            if b["stance"] != "ERROR" and p["stance"] == b["stance"]
        )
        comparable = sum(1 for b in results if b["stance"] != "ERROR")
        print(f"  {name:<26} {agree}/{comparable} pairs agree with pipeline stance")

    # Full output (untruncated reasoning, for manual coherence scoring)
    # Baseline 1 and 2 produce no reasoning text, so only pipeline vs
    # Baseline 3 are shown here.
    print(f"\n{'='*75}")
    print("FULL OUTPUT - PIPELINE vs BASELINE 3 (reasoning comparison)")
    print(f"{'='*75}")
    for p, b in zip(pipeline_results, baseline_results):
        print(f"\n{p['pair']} {p['timeframe']}")
        print(f"  PIPELINE  stance={p['stance']:<10} conf={p['confidence']:.3f}  "
              f"conflict={p['conflict']}")
        print(f"  Reasoning : {p['reasoning']}")
        print(f"  BASELINE3 stance={b['stance']:<10} conf={b['confidence']:.3f}  "
              f"conflict={b['conflict']}")
        print(f"  Reasoning : {b['reasoning']}")

    # Coherence scoring table - pipeline vs Baseline 3 only
    print(f"\n{'='*75}")
    print("COHERENCE SCORING - MANUAL STEP (pipeline vs Baseline 3 only)")
    print(f"{'='*75}")
    print("""
Score each reasoning output 1-5:
  5 - References specific signals and explains interactions clearly.
  4 - Relevant and mostly grounded. Minor gaps or vagueness.
  3 - Plausible but generic. Could apply to any market situation.
  2 - Partially contradicts signals or ignores key modalities.
  1 - Incoherent, factually wrong, or completely generic.
    """)
    print(f"{'='*75}")


# Main
def run():
    print(f"\n{'='*75}")
    print("ORCHESTRATOR EVALUATION - eval_orchestrator.py")
    print("Sections: Consistency Test | Baseline 1/2/3 Comparison")
    print(f"{'='*75}")

    print("\nInitialising orchestrator (loading FinBERT + YOLO)...")
    try:
        orch = Orchestrator()
        print("Ready.")
    except Exception as e:
        print(f"ERROR: Could not initialise orchestrator: {e}")
        print("Check FinBERT and YOLO are accessible and Ollama is running.")
        return

    print("Loading sentiment model for Baseline 2 (text-only)...")
    sentiment_pipe = load_sentiment_model()
    print("Ready.")

    run_consistency(orch)
    run_baseline_comparison(orch, sentiment_pipe)

    print(f"\n{'='*75}")
    print("EVALUATION COMPLETE")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    run()