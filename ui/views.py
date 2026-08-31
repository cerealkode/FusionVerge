import math
import streamlit as st
from core.feedback import load_weights, load_weight_history, save_weights, process_outcome_feedback
from infrastructure.database import update_outcome, fetch_history, fetch_analysis_by_id, fetch_history_count
from ui.constants import ASSETS, TIMEFRAMES, STANCE_CLASSES, CONFLICT_CLASSES, OUTCOME_LABELS
from ui.components import conf_bar_html, modality_card_html, weight_panel_html, history_row_html, weight_history_chart, conflict_detail_html


# Left section panels
def render_controls_panel() -> tuple[str, str, bool]:
    """Renders the asset selector, timeframe selector and run button.

    - Places dropdown inputs (pair/timeframe) and the execution trigger in the sidebar control panel
    - Keeps track of the selected pair across page refreshes using session state (st.session_state.selected_pair)

    Returns tuple of (pair, timeframe, run_btn_clicked).
    """
    st.markdown('<div class="section-label">Analysis Parameters</div>', unsafe_allow_html=True)

    # Asset pair selection dropdown
    pair = st.selectbox(
        "Asset",
        options=ASSETS,
        index=ASSETS.index(st.session_state.selected_pair),
        key="pair_select",
    )
    st.session_state.selected_pair = pair

    # Timeframe selector dropdown
    timeframe = st.selectbox(
        "Timeframe",
        options=TIMEFRAMES,
        format_func=lambda x: {"1h": "1 Hour", "1d": "Daily"}.get(x, x),
        key="tf_select",
    )

    run_disabled = st.session_state.running # disable the main run button when already running analysis pipeline
    run_btn = st.button(
        "RUN ANALYSIS" if not st.session_state.running else "RUNNING...",
        disabled=run_disabled,
        key="run_btn",
    )

    return pair, timeframe, run_btn

def render_history_panel(conn, pair: str) -> None:
    """Renders the historical list of past analyses for the active asset.

    - Uses fetch_history() for paginated rows and fetch_history_count() for page calculation
    - Loops through rows and builds consecutive summary cards as HTML snippets
    """
    # Reset back to page 1 whenever the selected asset changes
    if st.session_state.get("history_pair_for_paging") != pair:
        st.session_state.history_page = 0
        st.session_state.history_pair_for_paging = pair

    st.markdown(
        f'<div class="section-label">Recent - {pair}</div>',
        unsafe_allow_html=True,
    )

    # Grab history log from db
    # * combine entry into single joined block of inline html code
    page_size = 5
    total = fetch_history_count(conn, pair)
    offset = st.session_state.history_page * page_size
    history = fetch_history(conn, pair, limit=page_size, offset=offset)

    if history:
        for i, row in enumerate(history):
            row_col, btn_col = st.columns([12, 1], gap="small")
            with row_col:
                st.markdown(history_row_html(row, show_border=(i != len(history) - 1)), unsafe_allow_html=True)
            # Button to add outcome feedback
            with btn_col:
                if row["outcome"] is None:
                    with st.container(key=f"hist_btn_wrap_{row['id']}"):
                        if st.button("✎", key=f"hist_review_{row['id']}"):
                            select_history_row(conn, row["id"])
                            st.rerun()

        # Pagination controls
        total_pages = max(1, math.ceil(total / page_size))
        prev_col, label_col, next_col = st.columns([1, 2, 1], gap="small")
        with prev_col:
            with st.container(key="hist_page_btn_wrap_prev"):
                if st.button("← PREV", key="hist_prev_btn", disabled=st.session_state.history_page <= 0):
                    st.session_state.history_page -= 1
                    st.rerun()
        with label_col:
            st.markdown(
                f'<div class="feedback-note" style="text-align:center; margin-top:10px;">'
                f'PAGE {st.session_state.history_page + 1} / {total_pages}</div>',
                unsafe_allow_html=True,
            )
        with next_col:
            has_next = offset + len(history) < total
            with st.container(key="hist_page_btn_wrap_next"):
                if st.button("NEXT →", key="hist_next_btn", disabled=not has_next):
                    st.session_state.history_page += 1
                    st.rerun()
    else:
        st.markdown(
            '<div class="empty-state">NO ANALYSES YET FOR THIS ASSET</div>',
            unsafe_allow_html=True,
        )

def select_history_row(conn, row_id: int) -> None:
    """Reconstructs session state for a past run
    
    - So its outcome-feedback panel can be shown again on the right when giving outcome feedback
    - From DB data alone (no re-run of the pipeline)
    - Does NOT touch st.session_state.current_weights - weight panel on the left always reflects LIVE per-pair weights,
      not a snapshot from this past run
    """
    # Pulls full row from db
    row = fetch_analysis_by_id(conn, row_id)
    if row is None:
        return

    st.session_state.result = {
        "pair": row["pair"],
        "timeframe": row["timeframe"],
        "date": str(row["run_date"]),
        "stance": row["stance"],
        "confidence": row["confidence"],
        "conflict_level": row["conflict_level"],
        "conflict_imbalance": row["conflict_imbalance"], # may be None if dont exist
        "signals": row["signals"],
        "modality_outputs": row["modality_outputs"],
        "reasoning": row["reasoning"],
    }
    st.session_state.last_row_id = row_id
    st.session_state.feedback_submitted = row["outcome"] is not None
    st.session_state.feedback_message = None
    st.session_state.viewing_historical = True

def render_weight_state_panel(conn, pair: str) -> None:
    """Shows current per-pair modality weight multipliers as percentage bars.
    
    - Uses weights from the last analysis run (session state) so it reflects
      state it was actually used
    """
    st.markdown('<div class="section-label">Modality Weights</div>', unsafe_allow_html=True)

    display_weights = st.session_state.current_weights
    if display_weights is None:
        display_weights = load_weights(conn, pair) # no run yet this session, load from DB

    st.markdown(weight_panel_html(display_weights), unsafe_allow_html=True)

# Right section - fusion output
def render_fusion_output(conn, result: dict, last_row_id, is_historical: bool = False) -> None:
    """Renders the full analysis result: stance card, confidence bar, modality
    
    - signal cards, LLM reasoning, feedback, weight-trajectory chart and metadata strip at bottom
    """
    # Deconstruct dict keys for local-view variables
    stance = result["stance"]
    confidence = result["confidence"]
    conflict = result["conflict_level"]
    imbalance = result.get("conflict_imbalance")
    signals = result.get("signals", {})
    modality_outputs = result.get("modality_outputs", {})
    reasoning = result.get("reasoning", "")

    st.markdown('<div class="section-label">Fusion Output</div>', unsafe_allow_html=True)
    if is_historical:
        st.markdown(
            '<div class="feedback-note" style="margin:-6px 0 14px;">'
            'VIEWING A PAST RUN FROM HISTORY</div>',
            unsafe_allow_html=True,
        )

    # TOP level: Split side-by-side card layouts for fusion overview metrics
    # * left side = directional stance, right side = confidence
    s_col, c_col = st.columns([1, 1], gap="medium")
    with s_col:
        st.markdown(f"""
        <div class="stance-card {STANCE_CLASSES.get(stance, '')}">
            <div class="stance-label">{result['pair']} · {result['timeframe']} · {result['date']}</div>
            <div class="stance-value">{stance.upper()}</div>
            <div>
                <span class="conflict-badge {CONFLICT_CLASSES.get(conflict, '')}">
                    {conflict} conflict
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_col:
        st.markdown(conf_bar_html(confidence, stance), unsafe_allow_html=True)
        if imbalance is not None:
            st.markdown(conflict_detail_html(imbalance), unsafe_allow_html=True)
        else:
            # If row predates the conflict_imbalance column - add filler message instead of showing blank/null/0
            st.markdown(
                '<div class="conf-wrap" style="margin-top:16px;">'
                '<div class="conf-label"><span>CONFLICT DETAIL</span></div>'
                '<div class="empty-state" style="padding:8px 0;">'
                'NOT RECORDED FOR RUNS BEFORE THIS FEATURE WAS ADDED</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">Modality Signals</div>', unsafe_allow_html=True)

    # SECOND level: Modality cards (4), each in their block container
    mod_keys = ["news", "chart", "timeseries", "positioning"]
    cols = st.columns(4, gap="small")
    for i, key in enumerate(mod_keys):
        with cols[i]:
            mod_data = modality_outputs.get(key, {})
            summary = signals.get(key, "")
            # Render structured card (signal, conf, pattern) as HTML
            st.markdown(
                modality_card_html(key, mod_data),
                unsafe_allow_html=True,
            )
            # Description where avail for some cords
            if summary:
                st.caption(summary)

    # THIRD LEVEL: Reasoning chunk from LLM response
    st.markdown('<div class="section-label">Reasoning</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="reasoning-block">{reasoning}</div>',
        unsafe_allow_html=True,
    )

    # FOURTH level: show the feedback panel for users to add
    render_feedback_panel(conn)

    # FIFTH level: Weight trajectory chart
    # * hows how each modality's weight has evolved over the last N feedback event for this pair and timeframe
    # * only rendered if there is at least 1 feedback
    # * chart-building logic lives in ui.components.weight_history_chart()
    # * skipped for historical results as showing current weights may be misleading
    if not is_historical:
        st.markdown('<div class="section-label">Weight History</div>', unsafe_allow_html=True)
        weight_history = load_weight_history(conn, result["pair"], limit=20)
        if len(weight_history) < 2:
            st.markdown(
                '<div class="empty-state" style="padding:24px 0;">'
                'WEIGHT TRAJECTORY AVAILABLE AFTER FIRST FEEDBACK EVENT'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.altair_chart(weight_history_chart(weight_history), use_container_width=True)

    # Metadata line for beneath the chart
    st.markdown(
        f'<div class="meta-strip">'
        f'PAIR: {result["pair"]} · '
        f'TF: {result["timeframe"]} · '
        f'DATE: {result["date"]} · '
        f'DB ID: {last_row_id}'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_feedback_panel(conn) -> None:
    """Renders the outcome-feedback panel.
    
    - Radio selectors so the user can flag an output as correct, incorrect, uncertain
    - Automatically processes and updates pipeline weights in feedback loop based on inputs
    """
    if not (st.session_state.result and st.session_state.last_row_id):
        return

    st.markdown('<div class="section-label">Outcome Feedback</div>', unsafe_allow_html=True)

    # Create layout col to place radio options button side by side (save vertical space to avoid user need to scroll)
    radio_col, btn_col = st.columns([3, 1.3], gap="medium")
    with radio_col:
        outcome_choice = st.radio(
            "Was this analysis correct?",
            options=["correct", "incorrect", "uncertain"],
            format_func=lambda x: OUTCOME_LABELS[x],
            horizontal=True,
            key=f"outcome_radio_{st.session_state.last_row_id}",
            disabled=st.session_state.feedback_submitted,
        )
    with btn_col:
        # Align submit button with the radio options button
        st.markdown('<div style="margin-top: 26px;"></div>', unsafe_allow_html=True)
        submit_feedback = st.button(
            "SUBMIT OUTCOME",
            key="feedback_btn",
            disabled=st.session_state.feedback_submitted,
        )

    # Save user choice when submitted
    if submit_feedback and not st.session_state.feedback_submitted:
        update_outcome(
            conn,
            st.session_state.last_row_id,
            outcome_choice
        )

        # Compute and persist weight update
        # * decision logic of what update should be and what msg to show
        # * handle weight adjustment if model(s) was wrong
        # * pull existing multiplier, compute reward/penalty and log to db
        result_for_fb = st.session_state.result
        if result_for_fb:
            current_w = load_weights(conn, result_for_fb["pair"])
            updated_w, message = process_outcome_feedback(
                current_w,
                result_for_fb.get("modality_outputs", {}),
                outcome_choice,
                result_for_fb.get("stance", "neutral"),
            )
            if updated_w is not None:
                save_weights(conn, st.session_state.last_row_id, updated_w)
            st.session_state.current_weights = updated_w if updated_w is not None else current_w
        else:
            message = "Outcome recorded."

        # Lock form input and trigger immediate page refresh
        # * make buttons disabled and success banner to show up
        st.session_state.feedback_submitted = True
        st.session_state.feedback_message = message
        st.rerun()

    # Visual confirmation after submit
    if st.session_state.feedback_submitted:
        st.success(st.session_state.get("feedback_message", "Outcome recorded."))
        st.markdown(
            '<div class="feedback-note">✓ OUTCOME LOGGED</div>',
            unsafe_allow_html=True,
        )

def render_empty_state() -> None:
    """Renders the placeholder shown in the right column before any analysis has run.

    - Show that this space is 'reserved' for the fusion output and not dead space on the page
    """
    st.markdown("""
    <div class="output-panel">
        <div>
            <div class="output-panel-icon">◈</div>
            <div class="output-panel-text">SELECT AN ASSET AND TIMEFRAME<br>THEN RUN ANALYSIS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_loading_state(pair: str, timeframe: str) -> None:
    """Renders an interactive animated placeholder block during analysis pipeline loop.

    - Indicates that app is in running state and not crashed
    """
    st.markdown(f"""
    <div class="output-panel">
        <div>
            <div class="output-panel-icon spinning">◈</div>
            <div class="output-panel-text">RUNNING FULL PIPELINE<br>{pair.upper()} · {timeframe.upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)