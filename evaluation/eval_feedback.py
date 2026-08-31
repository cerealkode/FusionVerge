"""
Evaluation for feedback loop (core/feedback.py)

- To mainly answer hypothesis question 3:
    WEIGHT UPDATE LOGIC - verifies all documented weight update rules using predefined test cases.
    SIMULATED FEEDBACK TRAJECTORY - simulates sequential feedback events to visualise weight adaptation over time.
    CONSISTENCY COMPARISON - compares pipeline consistency before and after applying simulated feedback weights.
- Section 1 and 2 run fully offline. Section 3 requires Ollama, trained models in models/, and live API access

Usage - python evaluation/eval_feedback.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import statistics
import logging
import ollama
from core.feedback import compute_weight_update, apply_weights, weights_as_pct, MODALITIES
from infrastructure.config import DEFAULT_WEIGHT, LEARNING_RATE, WEIGHT_FLOOR, WEIGHT_CEILING

# Suppress INFO-level logging noise from core - this script's print() calls
# are the only output that should reach the console.
logging.getLogger().setLevel(logging.WARNING)
for _name in ("core", "modalities", "httpx", "ollama", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)


# Section 1 - Weight Update Logic Verification
UPDATE_CASES = [
    {
        "label": "Uncertain outcome - no update",
        "weights": {m: 1.0 for m in MODALITIES},
        "outputs": {m: {"signal": 1, "confidence": 0.7} for m in MODALITIES},
        "outcome": "uncertain", "stance": "bullish",
        "expected_none": True,
    },
    {
        "label": "All agree bullish + correct - all boosted",
        "weights": {m: 1.0 for m in MODALITIES},
        "outputs": {m: {"signal": 1, "confidence": 0.7} for m in MODALITIES},
        "outcome": "correct", "stance": "bullish",
        "expected_none": False,
        "expected": {m: "boost" for m in MODALITIES},
    },
    {
        "label": "All agree bullish + incorrect - no update (unanimous wrong)",
        "weights": {m: 1.0 for m in MODALITIES},
        "outputs": {m: {"signal": 1, "confidence": 0.7} for m in MODALITIES},
        "outcome": "incorrect", "stance": "bullish",
        "expected_none": True,
    },
    {
        "label": "3 bullish, 1 bearish dissent, correct - agreers boost, dissenter unchanged",
        "weights": {m: 1.0 for m in MODALITIES},
        "outputs": {
            "news":        {"signal": 1,  "confidence": 0.78},
            "chart":       {"signal": 1,  "confidence": 0.65},
            "timeseries":  {"signal": 1,  "confidence": 0.60},
            "positioning": {"signal": -1, "confidence": 0.62},
        },
        "outcome": "correct", "stance": "bullish",
        "expected_none": False,
        "expected": {
            "news": "boost", "chart": "boost", "timeseries": "boost",
            "positioning": "unchanged",
        },
    },
    {
        "label": "3 bullish, 1 bearish dissent, incorrect - agreers penalty, dissenter boost",
        "weights": {m: 1.0 for m in MODALITIES},
        "outputs": {
            "news":        {"signal": 1,  "confidence": 0.78},
            "chart":       {"signal": 1,  "confidence": 0.65},
            "timeseries":  {"signal": 1,  "confidence": 0.60},
            "positioning": {"signal": -1, "confidence": 0.62},
        },
        "outcome": "incorrect", "stance": "bullish",
        "expected_none": False,
        "expected": {
            "news": "penalty", "chart": "penalty", "timeseries": "penalty",
            "positioning": "boost",
        },
    },
    {
        "label": "Neutral signal modality - unchanged regardless of outcome",
        "weights": {m: 1.0 for m in MODALITIES},
        "outputs": {
            "news":        {"signal": 1,  "confidence": 0.78},
            "chart":       {"signal": 1,  "confidence": 0.65},
            "timeseries":  {"signal": 0,  "confidence": 0.50},
            "positioning": {"signal": -1, "confidence": 0.62},
        },
        "outcome": "incorrect", "stance": "bullish",
        "expected_none": False,
        "expected": {
            "news": "penalty", "chart": "penalty",
            "timeseries": "unchanged", "positioning": "boost",
        },
    },
    {
        "label": "Floor enforcement - near-floor weight not pushed below WEIGHT_FLOOR",
        "weights": {m: WEIGHT_FLOOR + 0.05 for m in MODALITIES},
        "outputs": {
            "news":        {"signal": 1,  "confidence": 0.7},
            "chart":       {"signal": 1,  "confidence": 0.7},
            "timeseries":  {"signal": 1,  "confidence": 0.7},
            "positioning": {"signal": -1, "confidence": 0.7},
        },
        "outcome": "incorrect", "stance": "bullish",
        "expected_none": False,
        "check_floor": ["news", "chart", "timeseries"],
    },
    {
        "label": "Ceiling enforcement - near-ceiling weight not pushed above WEIGHT_CEILING",
        "weights": {m: WEIGHT_CEILING - 0.05 for m in MODALITIES},
        "outputs": {m: {"signal": 1, "confidence": 0.7} for m in MODALITIES},
        "outcome": "correct", "stance": "bullish",
        "expected_none": False,
        "check_ceiling": list(MODALITIES),
    },
]

def run_weight_update_verification():
    print(f"\n{'='*78}")
    print("SECTION 1 - WEIGHT UPDATE LOGIC VERIFICATION")
    print(f"{'='*78}\n")

    passed = 0
    failed = 0

    for i, case in enumerate(UPDATE_CASES, start=1):
        result = compute_weight_update(
            case["weights"], case["outputs"], case["outcome"], case["stance"]
        )
        status = "PASS"
        notes = []

        if case.get("expected_none"):
            if result is not None:
                status = "FAIL"
                notes.append(f"expected None, got {result}")
        else:
            if result is None:
                status = "FAIL"
                notes.append("expected a dict, got None")
            else:
                for mod, expectation in case.get("expected", {}).items():
                    before = case["weights"][mod]
                    after = result[mod]
                    if expectation == "boost" and after <= before:
                        status = "FAIL"
                        notes.append(f"{mod}: expected boost ({before:.3f} -> {after:.3f})")
                    elif expectation == "penalty" and after >= before:
                        status = "FAIL"
                        notes.append(f"{mod}: expected penalty ({before:.3f} -> {after:.3f})")
                    elif expectation == "unchanged" and round(after, 6) != round(before, 6):
                        status = "FAIL"
                        notes.append(f"{mod}: expected unchanged ({before:.3f} -> {after:.3f})")
                for mod in case.get("check_floor", []):
                    if result[mod] < WEIGHT_FLOOR:
                        status = "FAIL"
                        notes.append(f"{mod}: {result[mod]:.4f} below WEIGHT_FLOOR")
                for mod in case.get("check_ceiling", []):
                    if result[mod] > WEIGHT_CEILING:
                        status = "FAIL"
                        notes.append(f"{mod}: {result[mod]:.4f} above WEIGHT_CEILING")

        label_trunc = case["label"][:60].ljust(60)
        print(f"  {i}. {label_trunc} [{status}]")
        for note in notes:
            print(f"     FAIL: {note}")

        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print(f"\n  Result: {passed}/{passed + failed} passed")
    print(f"{'='*78}")


# Section 2 - Simulated Feedback Trajectory
SIMULATION_EVENTS = [
    {"signals": {"news":  1, "chart":  1, "timeseries":  1, "positioning": -1}, "stance": "bullish",  "outcome": "correct"},
    {"signals": {"news":  1, "chart": -1, "timeseries":  1, "positioning":  1}, "stance": "bullish",  "outcome": "correct"},
    {"signals": {"news": -1, "chart": -1, "timeseries": -1, "positioning": -1}, "stance": "bearish",  "outcome": "correct"},
    {"signals": {"news":  1, "chart":  1, "timeseries": -1, "positioning":  1}, "stance": "bullish",  "outcome": "incorrect"},
    {"signals": {"news": -1, "chart": -1, "timeseries":  1, "positioning": -1}, "stance": "bearish",  "outcome": "correct"},
    {"signals": {"news":  1, "chart":  1, "timeseries":  1, "positioning":  1}, "stance": "bullish",  "outcome": "correct"},
    {"signals": {"news": -1, "chart":  1, "timeseries": -1, "positioning": -1}, "stance": "bearish",  "outcome": "incorrect"},
    {"signals": {"news":  1, "chart":  1, "timeseries":  0, "positioning":  1}, "stance": "bullish",  "outcome": "correct"},
    {"signals": {"news": -1, "chart": -1, "timeseries": -1, "positioning":  1}, "stance": "bearish",  "outcome": "correct"},
    {"signals": {"news":  1, "chart": -1, "timeseries": -1, "positioning":  1}, "stance": "neutral",  "outcome": "uncertain"},
    {"signals": {"news":  1, "chart":  1, "timeseries":  1, "positioning": -1}, "stance": "bullish",  "outcome": "correct"},
    {"signals": {"news": -1, "chart": -1, "timeseries":  1, "positioning": -1}, "stance": "bearish",  "outcome": "correct"},
    {"signals": {"news":  1, "chart":  1, "timeseries": -1, "positioning":  1}, "stance": "bullish",  "outcome": "incorrect"},
    {"signals": {"news": -1, "chart": -1, "timeseries": -1, "positioning": -1}, "stance": "bearish",  "outcome": "correct"},
    {"signals": {"news":  1, "chart":  1, "timeseries":  1, "positioning":  1}, "stance": "bullish",  "outcome": "correct"},
]


def run_trajectory_simulation():
    print(f"\n{'='*78}")
    print("SECTION 2 - SIMULATED FEEDBACK TRAJECTORY (15 events)")
    print(f"{'='*78}\n")

    weights = {m: DEFAULT_WEIGHT for m in MODALITIES}
    trajectory = [dict(weights)]
    skip_count = 0
    update_count = 0

    for i, event in enumerate(SIMULATION_EVENTS, start=1):
        outputs = {
            m: {"signal": event["signals"][m], "confidence": 0.70}
            for m in MODALITIES
        }
        result = compute_weight_update(
            weights, outputs, event["outcome"], event["stance"]
        )
        if result is not None:
            weights = result
            update_count += 1
        else:
            skip_count += 1
        trajectory.append(dict(weights))

    # Trajectory table
    col = 12
    print(f"{'Event':<8}", end="")
    for m in MODALITIES:
        print(f"{m[:col-1]:<{col}}", end="")
    print()
    print("-" * (8 + col * len(MODALITIES)))

    for i, snap in enumerate(trajectory):
        label = "Init" if i == 0 else str(i)
        print(f"{label:<8}", end="")
        for m in MODALITIES:
            val = snap[m]
            if i > 0:
                prev = trajectory[i - 1][m]
                marker = "▲" if val > prev else ("▼" if val < prev else " ")
            else:
                marker = " "
            print(f"{val:.3f}{marker:<{col-5}}", end="")
        print()

    # Final state summary
    print(f"\n{'─'*78}")
    print(f"{'Modality':<15} {'Initial':<10} {'Final':<10} {'Delta':<10} {'% share'}")
    print("-" * 55)
    pcts = weights_as_pct(weights)
    for m in MODALITIES:
        delta = weights[m] - DEFAULT_WEIGHT
        print(f"{m:<15} {DEFAULT_WEIGHT:<10.3f} {weights[m]:<10.3f} {delta:+.3f}{'':6} {pcts[m]}%")

    print(f"\n  Updates applied : {update_count} / {len(SIMULATION_EVENTS)}")
    print(f"  Skipped (no-op) : {skip_count}")
    print(f"  (skipped = uncertain outcome or unanimous-wrong events)")
    print(f"{'='*78}")

    return weights


# Section 3 - Consistency Comparison (requires Ollama + models)
CONSISTENCY_RUNS = 10
CONSISTENCY_PAIR = "EURUSD"
CONSISTENCY_TF = "1h"

def run_consistency_comparison(weights_after_feedback: dict):
    print(f"\n{'='*78}")
    print("SECTION 3 - CONSISTENCY COMPARISON")
    print(f"Default weights vs post-feedback weights | {CONSISTENCY_RUNS} runs each")
    print(f"{'='*78}\n")

    try:
        ollama.list()
    except Exception:
        print("SKIPPED - Ollama not reachable. Start Ollama with 'ollama serve'.")
        return

    from infrastructure.config import GRU_BEST_CONFIGS_PATH
    if not (os.path.exists(GRU_BEST_CONFIGS_PATH) and
            os.path.exists(f"models/{CONSISTENCY_PAIR}_{CONSISTENCY_TF}.keras")):
        print("SKIPPED - Trained model files not found. Run training notebook first.")
        return

    from core.orchestrator import Orchestrator

    orch = Orchestrator()

    def run_n(label: str, weights: dict) -> list[dict]:
        print(f"\n  [{label}]")
        results = []
        for i in range(1, CONSISTENCY_RUNS + 1):
            try:
                raw = orch.run(CONSISTENCY_PAIR, CONSISTENCY_TF)
                # Manually apply simulated weights for the post-feedback run
                if label != "DEFAULT":
                    raw["modality_outputs"] = apply_weights(
                        raw["modality_outputs"], weights
                    )
                results.append({
                    "run": i,
                    "stance": raw["stance"],
                    "confidence": raw["confidence"],
                    "conflict_level": raw["conflict_level"],
                })
                print(f"    Run {i}: {raw['stance']:<10} conf={raw['confidence']:.3f} "
                      f"conflict={raw['conflict_level']}")
            except Exception as e:
                print(f"    Run {i}: ERROR - {e}")
        return results

    default_w = {m: DEFAULT_WEIGHT for m in MODALITIES}
    base = run_n("DEFAULT", default_w)
    fb = run_n("POST-FEEDBACK", weights_after_feedback)

    def summarise(results):
        valid = [r for r in results if "stance" in r]
        if not valid:
            return None
        counts = {}
        for r in valid:
            counts[r["stance"]] = counts.get(r["stance"], 0) + 1
        top = max(counts, key=counts.get)
        confs = [r["confidence"] for r in valid]
        return {
            "rate": counts[top] / len(valid) * 100,
            "top": top,
            "mean": statistics.mean(confs),
            "stdev": statistics.stdev(confs) if len(confs) > 1 else 0.0,
        }

    bs = summarise(base)
    fs = summarise(fb)

    print(f"\n{'='*78}")
    print("COMPARISON")
    print(f"{'='*78}")

    if bs and fs:
        print(f"\n  {'Metric':<30} {'Default':<20} {'Post-Feedback'}")
        print("  " + "-" * 65)
        print(f"  {'Consistency rate':<30} {bs['rate']:.0f}%{'':<16}{fs['rate']:.0f}%")
        print(f"  {'Most common stance':<30} {bs['top']:<20}{fs['top']}")
        print(f"  {'Confidence mean':<30} {bs['mean']:.3f}{'':<17}{fs['mean']:.3f}")
        print(f"  {'Confidence std dev':<30} {bs['stdev']:.3f}{'':<17}{fs['stdev']:.3f}")

    print(f"{'='*78}")


# Main
def run():
    print(f"\n{'='*78}")
    print("FEEDBACK EVALUATION - eval_feedback.py")
    print(f"{'='*78}")
    run_weight_update_verification()
    final_weights = run_trajectory_simulation()
    run_consistency_comparison(final_weights)
    print(f"\n{'='*78}")
    print("EVALUATION COMPLETE")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    run()