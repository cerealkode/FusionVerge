import pandas as pd
import altair as alt
from core.feedback import weights_as_pct
from ui.constants import SIGNAL_LABELS, SIGNAL_CLASSES, CONFLICT_CLASSES, MODALITY_NAMES, OUTCOME_LABELS, OUTCOME_CLASSES, PATTERN_LABELS, CONFLICT_LOW_THRESHOLD, CONFLICT_MEDIUM_THRESHOLD

def conf_bar_html(confidence: float, stance: str) -> str:
    """Renders the confidence progress bar as inline HTML.

    - Converts the confidence float into a clean percentage string for the UI
    - Colors the progress bar based on whether the direction is bullish, bearish or neutral

    Returns an HTML string containing the structured confidence display.
    """
    # Calculate percentage and grab theme colors for the active trading stance
    # * Uses CSS custom variables defined in our global dashboard stylesheet
    pct = round(confidence * 100)
    colour_map = {"bullish": "var(--bull)", "bearish": "var(--bear)", "neutral": "var(--neutral)"}
    colour = colour_map.get(stance, "var(--neutral)")
    return f"""
    <div class="conf-wrap">
        <div class="conf-label">
            <span>CONFIDENCE
                <span class="info-icon">&#9432;
                    <span class="tooltip-text">
                        The model's own self-reported confidence in this stance (0-100%).
                        This is separate from modality agreement - see CONFLICT DETAIL below.
                    </span>
                </span>
            </span>
            <span>{pct}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill" style="width:{pct}%; background:{colour};"></div>
        </div>
    </div>
    """

def conflict_detail_html(imbalance: float) -> str:
    """Renders a small panel showing the raw imbalance value and bucket boundaries.

    - Sits below conf_bar_html() in c_col so it fills the empty space to match stance-card's height
    """
    pct = round(imbalance * 100)
    low_pct = int(CONFLICT_LOW_THRESHOLD * 100)
    med_pct = int(CONFLICT_MEDIUM_THRESHOLD * 100)
    return f"""
    <div class="conf-wrap" style="margin-top:16px;">
        <div class="conf-label">
            <span>CONFLICT DETAIL
                <span class="info-icon">&#9432;
                    <span class="tooltip-text">
                        Imbalance measures how strongly the four modalities agreed on direction
                        (bull vs bear confidence), shown as 0-100%. Closer to 100% = stronger
                        agreement = lower conflict.<br><br>
                        LOW: above {low_pct}%<br>
                        MEDIUM: {med_pct}-{low_pct}%<br>
                        HIGH: below {med_pct}%
                    </span>
                </span>
            </span>
            <span>{pct}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill" style="width:{pct}%; background:var(--label);"></div>
        </div>
    </div>
    """

def modality_card_html(key: str, data: dict) -> str:
    """Renders the structured part of a modality signal card as inline HTML.

    - Extracts the directional signal, confidence score and weight multiplier
    - Handles feedback loop states by showing raw versus effective confidence separately (if weights change)
    - Keeps the summary text separate so Streamlit doesnt break over random text characters

    Returns an HTML string for a single component card.
    """
    # Pull data fields out of the modality dictionary object
    # * Returns neutral or zero fallbacks if any expected metrics are missing
    sig = data.get("signal", 0)
    conf = data.get("confidence", 0.0) # effective (post-weight)
    raw_conf = data.get("raw_confidence") # none if weights not yet applied
    weight = data.get("weight")
    label = SIGNAL_LABELS.get(sig, "NEUTRAL")
    cls = SIGNAL_CLASSES.get(sig, "sig-neut")
    name = MODALITY_NAMES.get(key, key.upper())

    extra = ""
    # Inject pattern labels if the active component is the chart modality
    # * convert any code string into a readable label for display,
    #   eg. M_Head = Double Top (M-Head). as shown in PATTERN_LABELS in constants.py
    # * the code string are outputs directly from the YOLO model 'supported labels' section - https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8
    if key == "chart" and data.get("pattern") and data["pattern"] != "none":
        pattern_label = PATTERN_LABELS.get(data["pattern"], data["pattern"].upper())
        extra += f'<div class="mod-conf">PATTERN: {pattern_label}</div>'

    # Show weight info only when a non-default weight is present
    # * keep it short and tight to have it as a short description in the card
    # * show how the feedback history multiplier is affecting the modal
    if weight is not None and raw_conf is not None and abs(weight - 1.0) > 0.001:
        extra += (
            f'<div class="mod-conf" style="color:var(--accent);">'
            f'RAW: {round(raw_conf * 100)}% · ×{weight:.3f} → EFF: {round(conf * 100)}%'
            f'</div>'
        )
    else:
        extra += f'<div class="mod-conf">CONF: {round(conf * 100)}%</div>'

    return f"""
    <div class="mod-card">
        <div class="mod-name">{name}</div>
        <div class="mod-signal {cls}">{label}</div>
        {extra}
    </div>
    """

def weight_panel_html(weights: dict) -> str:
    """Renders the current per-pair weight state as a horizontal tracking bar panel.

    - Shows each modality as a percentage of total influence (intuitive),
      alongside the raw multiplier value (precise)
    - eg. news=1.1 displays as 29% with ×1.100 label

    Returns an HTML string displaying the complete pipeline weight configuration.
    """
    pcts = weights_as_pct(weights)
    rows = ""
    names = {
        "news": "NEWS",
        "chart": "CHART",
        "timeseries": "TIME-SERIES",
        "positioning": "POSITIONING",
    }
    for mod in ["news", "chart", "timeseries", "positioning"]:
        pct = pcts.get(mod, 25)
        mult = weights.get(mod, 1.0)
        rows += f"""
        <div class="weight-row">
            <span class="weight-name">{names[mod]}</span>
            <div class="weight-bar-track">
                <div class="weight-bar-fill" style="width:{pct}%;"></div>
            </div>
            <span class="weight-pct">{pct}%</span>
            <span class="weight-mult">×{mult:.3f}</span>
        </div>"""
    return f'<div class="weight-panel">{rows}</div>'

def weight_history_chart(weight_history: list) -> alt.Chart:
    """Builds the weight-trajectory line chart shown after the first feedback.

    - Provides chart as visualization for user to see past weights adjustment
      based off previous feedbacks
    - Source for overall usage from example 3 - https://www.geeksforgeeks.org/python/introduction-to-altair-in-python/,
      difference is i added more beautification to it
    
    Returns an Altair Chart object ready for st.altair_chart().
    """
    # Build tidy df for Altair, one row per (timestamp, modality)
    rows = []
    for i, entry in enumerate(weight_history):
        for mod, val in entry["weights"].items():
            rows.append({
                "event": i + 1,
                "modality": mod.upper(),
                "weight": val,
            })
    df_w = pd.DataFrame(rows)

    mod_colours = {
        "NEWS": "blue",
        "CHART": "seagreen",
        "TIMESERIES": "orange",
        "POSITIONING": "crimson",
    }

    chart = (
        alt.Chart(df_w).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("event:O", title="Feedback Event",
                     axis=alt.Axis(labelColor="#a3a3c9", titleColor="#a3a3c9",
                                   labelAngle=0)),
            y=alt.Y("weight:Q", title="Weight Multiplier",
                     scale=alt.Scale(domain=[0.0, 2.1]),
                     axis=alt.Axis(labelColor="#a3a3c9", titleColor="#a3a3c9",
                                   titlePadding=10)),
            color=alt.Color(
                "modality:N",
                scale=alt.Scale(
                    domain=list(mod_colours.keys()),
                    range=list(mod_colours.values()),
                ),
                # Legends for the color of the modalitites
                legend=alt.Legend(
                    orient="bottom",
                    title="MODALITY LEGEND:",
                    titleOrient="left",
                    titlePadding=8,
                    titleFontSize=10,
                    labelFontSize=10,
                    labelColor="#a3a3c9",
                    titleColor="#a3a3c9",
                    symbolSize=80,
                ),
            ),
            tooltip=["event:O", "modality:N", "weight:Q"],
        )
        .properties(
            height=220,
            width="container",
            background="#111118",
            padding={"left": 12, "top": 10, "right": 10, "bottom": 5},
        )
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#1e1e2e", domainColor="#1e1e2e")
    )
    return chart

def history_row_html(row: dict, show_border: bool = True) -> str:
    """Renders a single history row as inline HTML."""
    # Fetch historical data fields and resolve signal class names
    # * map bull/bear directional number directly to the styling tags
    stance = row["stance"]
    cls = SIGNAL_CLASSES.get(
        1 if stance == "bullish" else (-1 if stance == "bearish" else 0), "sig-neut"
    )
    # Process the ground-truth verification outcome labels
    # * if user did not give feedback, just set status to PENDING
    outcome = row.get("outcome")
    outcome_html = ""
    if outcome:
        oc = OUTCOME_CLASSES.get(outcome, "outcome-pending")
        ol = OUTCOME_LABELS.get(outcome, outcome.upper())
        outcome_html = f'<span class="hist-outcome {oc}">{ol}</span>'
    else:
        outcome_html = '<span class="hist-outcome outcome-pending">PENDING</span>'

    # Timestamp object into human readable format
    # * for showing the timestamp of each previous analysis
    ts = row["run_at"]
    if hasattr(ts, "strftime"):
        ts_str = ts.strftime("%d %b %H:%M") # not using years as takes up too much space and note really that important
    else:
        ts_str = str(ts)[:16]

    conf_pct = round(row["confidence"] * 100)
    conflict = row["conflict_level"]
    conflict_cls = CONFLICT_CLASSES.get(conflict, "")

    border_style = "border-bottom:1px solid var(--border);" if show_border else ""
    return f"""
    <div class="hist-row" style="{border_style}">
        <span class="hist-tf">{row['timeframe']}</span>
        <span class="hist-stance {cls}">{stance.upper()}</span>
        <span class="hist-conf">{conf_pct}%</span>
        <span class="hist-conflict"><span class="conflict-badge {conflict_cls}">{conflict}</span></span>
        {outcome_html}
        <span class="hist-time">{ts_str}</span>
    </div>
    """