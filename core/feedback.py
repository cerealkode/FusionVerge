import logging
import json
from infrastructure.config import DEFAULT_WEIGHT, LEARNING_RATE, WEIGHT_FLOOR, WEIGHT_CEILING, MODALITIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Weight retrieval
def load_weights(conn, pair: str) -> dict:
    """Load latest per-pair modality weights from DB.

    - Fetch most recent weight snapshot from weight_log table joined via feedback_log
    - Falls back to DEFAULT_WEIGHT (1.0) for all modalities if no history exists (expected for first run)

    Returns dict with modality keys: news, chart, timeseries, positioning.
    """
    # Loads latest saved weights for pair
    with conn.cursor() as cur:
        cur.execute("""
            SELECT wl.weights
            FROM weight_log wl
            JOIN feedback_log fl ON wl.feedback_id = fl.id
            WHERE fl.pair = %s
            ORDER BY wl.logged_at DESC
            LIMIT 1
        """, (pair,))
        row = cur.fetchone()

    # If no saved weights (first ever run), then jsut use default weight of 1.0 ea
    if row is None:
        logger.info(f"No weight history for {pair}, using defaults.")
        return {m: DEFAULT_WEIGHT for m in MODALITIES}

    weights = row[0]
    # Ensure all modalities exists (backward compatibility if schema changed)
    for m in MODALITIES:
        if m not in weights:
            weights[m] = DEFAULT_WEIGHT

    logger.info(f"Loaded weights for {pair}: {weights}")
    return weights

def load_weight_history(conn, pair: str, limit: int = 20) -> list[dict]:
    """Load the historical weight trajectory for a pair

    - To render the weight trajectory chart in the dashboard,
      and show/visualize how weightage changed
    - Each snapshot entry correspond to 1 feedback event

    Returns list of dicts with keys: logged_at, weights (dict), feedback_id.
    """
    # Get weight history for pair, oldest to newest
    with conn.cursor() as cur:
        cur.execute("""
            SELECT wl.logged_at, wl.weights, wl.feedback_id
            FROM weight_log wl
            JOIN feedback_log fl ON wl.feedback_id = fl.id
            WHERE fl.pair = %s
            ORDER BY wl.logged_at ASC
            LIMIT %s
        """, (pair, limit))
        rows = cur.fetchall()

    # Each row/record will be 1 learning step after feedback
    history = [
        {"logged_at": r[0], "weights": r[1], "feedback_id": r[2]}
        for r in rows
    ]
    logger.info(f"Loaded {len(history)} weight history entries for {pair}.")
    return history

# Weight application
def apply_weights(modality_outputs: dict, weights: dict) -> dict:
    """Applies weight multipliers to raw confidence scores before fusion.

    - Weight acts as learned trust multiplier from feedback loop,
      so 'learned stuff' actually influence decision
    - effective_confidence = raw_confidence * weight,
      clamped to [0.0, 1.0]
    - effective confidence overwrites for downstream FUSION only, to see adjusted value
    - raw confidence is still kept for transparency

    Returns new modality_outputs dict with effective_confidence added per modality.
    """
    weighted = {}
    for mod, data in modality_outputs.items():
        w = weights.get(mod, DEFAULT_WEIGHT)
        # Raw model output confidence (before learning adjustment)
        raw_conf = data.get("confidence", 0.0)

        # Apply learned reliability scaling
        effective_conf = round(min(raw_conf * w, 1.0), 4) # calmped to prevent overflow
        weighted[mod] = {
            **data,
            "raw_confidence": raw_conf, # original conf
            "weight": round(w, 4), # trust multiplier for adjustment
            "confidence": effective_conf, # adjusted effective conf, what FUSION sees
        }
        logger.debug(
            f"{mod}: raw_conf={raw_conf:.4f} weight={w:.4f} "
            f"effective={effective_conf:.4f}"
        )
    return weighted

# Weight update logic
def compute_weight_update(weights: dict, modality_outputs: dict, outcome: str, stance: str) -> dict | None:
    """Computes updated weights after a user feedback event.

    - Core learning mechanism, adjusting modality trust based on real results (from feedback)
    - Result | Modality agreement with final fused stance
      Correct + agree -> small reward
      Correct + disagree -> no change (avoid penalising cautious/opposing signal if end result is fine)
      Wrong + agree -> penalty
      Wrong + disagree -> reward (they were right against the consensus)
    - Skip if outcome is uncertain, or all modalities agree to stance but outcome is wrong

    Returns updated weights dict or None (if no update applied).
    """
    if outcome == "uncertain":
        logger.info("Outcome uncertain - no weight update.")
        return None

    # Map stance string to signal int for comparison
    stance_signal = {"bullish": 1, "bearish": -1, "neutral": 0}.get(stance, 0)

    # Check if all non-neutral modalities agreed (since we dont care about uncertained neutrals)
    non_neutral = [
        m for m, d in modality_outputs.items()
        if d.get("signal", 0) != 0
    ]

    all_agreed = all(
        modality_outputs[m].get("signal", 0) == stance_signal
        for m in non_neutral
    ) if non_neutral else True # if empty list (neutrals), take it as all agree

    if outcome == "incorrect" and all_agreed:
        logger.info(
            "All modalities agreed on wrong outcome - no differential signal to attribute, skipping update."
        )
        return None

    updated = weights.copy()
    for mod in MODALITIES:
        sig = modality_outputs.get(mod, {}).get("signal", 0)
        agreed = (sig == stance_signal) and (stance_signal != 0) # whether modality supported the final stance
        
        # Reward correct contributors
        if outcome == "correct" and agreed:
            updated[mod] = round(min(weights[mod] + LEARNING_RATE, WEIGHT_CEILING), 3)
        # Penalise wrong decision contributors
        elif outcome == "incorrect" and agreed:
            updated[mod] = round(max(weights[mod] - LEARNING_RATE, WEIGHT_FLOOR), 3)
        # Reward dessenter (when modality disagrees with FUSION final decision that turns out to be wrong stance)
        elif outcome == "incorrect" and not agreed and sig != 0:
            updated[mod] = round(min(weights[mod] + LEARNING_RATE, WEIGHT_CEILING), 3)
        # correct + disagreed: no change (avoid pusnishing cautious signal)
        # neutral signal: no change regardless of outcome as no learning signal

    logger.info(f"Weight update: before={weights} after={updated}")
    return updated

def process_outcome_feedback(current_weights: dict, modality_outputs: dict, outcome: str, stance: str) -> tuple[dict | None, str]:
    """Handle user's feedback event and decide if weight update should occur.
    
    - A control layer before compute_weight_update()
    - Decides if update shold happen and return user-facing messag for the dashboard

    Returns tuple of (new_weights_or_None, status_message).
    """
    # Uncertainty = skip
    if outcome == "uncertain":
        return None, "Outcome recorded."

    # Calls the actual update logic for rule handling
    updated = compute_weight_update(current_weights, modality_outputs, outcome, stance)
    if updated is not None:
        return updated, "Outcome recorded. Weights updated." # reflect if weights changed
    return None, "Outcome recorded. No weight change (unanimous agreement)."

def save_weights(conn, feedback_id: int, weights: dict) -> None:
    """Persist updated weights snapshot in DB.
    
    - Learning and feedback should persist across all runs
    - This also allow for trajectory of weights as visualization for users and evaluation
    """
    # Save updated weights linked to the feedback event
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO weight_log (feedback_id, weights)
            VALUES (%s, %s)
        """, (feedback_id, json.dumps(weights)))
    logger.info(f"Weights saved for feedback_id={feedback_id}: {weights}")

def weights_as_pct(weights: dict) -> dict:
    """Converts raw multiplier weights to percentage-of-total for display.
    
    - Example of {news: 1.1, chart: 0.9, timeseries: 1.0, positioning: 1.0} becomes:
      {news: 28, chart: 22, timeseries: 25, positioning: 25}
    - Only for UI presentation sake

    Returns dict with same keys (modality name) and the integer percentage values.
    """
    total = sum(weights.values())

    # Edge case of when all total is 0, we just split it evenly
    if total == 0:
        return {m: 25 for m in MODALITIES}
    return {
        # Should more of less sum up to 100 plus minus 1
        m: round((v / total) * 100)
        for m, v in weights.items()
    }


# Run smoke test with python -m core.feedback
if __name__ == "__main__":
    # requires a live DB connection
    import os
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True

    test_weights = {"news": 1.0, "chart": 1.0, "timeseries": 1.0, "positioning": 1.0}
    test_outputs = {
        "news": {"signal": 1,  "confidence": 0.72},
        "chart": {"signal": -1, "confidence": 0.35},
        "timeseries": {"signal": 1,  "confidence": 0.60},
        "positioning": {"signal": 1,  "confidence": 0.62},
    }

    updated = compute_weight_update(test_weights, test_outputs, "incorrect", "bullish")
    print(f"Updated weights: {updated}")
    print(f"As percentages: {weights_as_pct(updated or test_weights)}")
    conn.close()