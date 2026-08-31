# Mute warning logs (not errors) to avoid clutter
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv
from core.orchestrator import Orchestrator
from ui.styles import load_css
from ui.views import render_controls_panel, render_history_panel, render_weight_state_panel, render_fusion_output, render_empty_state, render_loading_state, select_history_row
from infrastructure.database import get_db_connection, init_db, insert_analysis

load_dotenv()


# Page config, needs to be first Streamlit call
st.set_page_config(
    page_title="FusionVerge",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load in stylings
load_css()

# Load orchestrator for entire pipeline (once per session)
@st.cache_resource(show_spinner=False)
def get_orchestrator():
    """Loads FinBERT and YOLO once per Streamlit session.

    - st.cache_resource persist across rerenders so models are not reloaded everytime
    """
    return Orchestrator()

# Session state initialisation
if "result" not in st.session_state:
    st.session_state.result = None
if "last_row_id" not in st.session_state:
    st.session_state.last_row_id = None
if "running" not in st.session_state:
    st.session_state.running = False
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = None
if "selected_pair" not in st.session_state:
    st.session_state.selected_pair = "EURUSD"
if "current_weights" not in st.session_state:
    st.session_state.current_weights = None # populated after analysis or feedback
if "viewing_historical" not in st.session_state:
    st.session_state.viewing_historical = False # True when result was reloaded from a past history row (see ui.views.select_history_row)
if "history_page" not in st.session_state:
    st.session_state.history_page = 0
if "history_pair_for_paging" not in st.session_state:
    st.session_state.history_pair_for_paging = None


# DB init
conn = get_db_connection()
init_db(conn)


# Click-to-feedback from History (iteration 2 change)
# * handle ?review_id=<id> from the HTML history table, then clear it after use
# * allow user to add outcome feedback to old analysis
review_id = st.query_params.get("review_id")
if review_id:
    select_history_row(conn, int(review_id))
    st.query_params.clear()
    st.rerun()


# Header
st.markdown("""
<div class="fv-header">
    <span class="fv-title">◈ FusionVerge</span>
</div>
""", unsafe_allow_html=True)



# Layout
# * left contains pair/tf selector, history, modalities weightage
# * right contains outputs after analysis made
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    pair, timeframe, run_btn = render_controls_panel()
    render_history_panel(conn, pair)
    render_weight_state_panel(conn, pair)


# Pipeline execution
# * left in main.py instead of views.py as st.session_state-driven controls flow rather than render
with right_col:
    if run_btn:
        st.session_state.running = True
        st.session_state.result = None
        st.session_state.last_row_id = None
        st.session_state.feedback_submitted = False
        st.session_state.viewing_historical = False
        st.rerun()

    if st.session_state.running and st.session_state.result is None:
        # Rendering of the 'loading' state during analysis
        # * streamlit flush this markdown to the browser before the blocking orch.run() call below executes
        render_loading_state(pair, timeframe)

        try:
            orch = get_orchestrator()
            # Inject live DB connection so orchestrator can load per-pair weights
            orch.db_conn = conn
            result = orch.run(pair, timeframe)
            st.session_state.result = result
            st.session_state.last_row_id = insert_analysis(conn, result)
            # Store weights used for this run so the weight panel reflects state used to feed this analysis
            st.session_state.current_weights = result.get("weights")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.session_state.result = None
        finally:
            st.session_state.running = False
        st.rerun()

    result = st.session_state.result

    if result is None:
        render_empty_state()
    else:
        render_fusion_output(conn, result, st.session_state.last_row_id, st.session_state.viewing_historical)