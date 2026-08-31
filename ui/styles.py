import streamlit as st

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

    :root {
        --bg:        #0a0a0f;
        --surface:   #111118;
        --border:    #1e1e2e;
        --accent:    #c8f535;
        --bull:      #39d98a;
        --bear:      #ff4d6d;
        --neutral:   #8888aa;
        --text:      #e8e8f0;
        --muted:     #555570;
        --label:     #a3a3c9;  /* brighter than --muted for section headers,
                                  card labels and anything meant to be read,
                                  else hard to see due to dark bg */         
        --conf-bg:   #1a1a28;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Syne', sans-serif !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: var(--bg) !important;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    /* visibility:hidden above still reserves the header's height, which is
       what was pushing 'Page title' down the page - collapse it fully */
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem !important;
    }

    /* Typography */
    h1, h2, h3, h4 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
    }

    /* Header bar */
    .fv-header {
        display: flex;
        align-items: baseline;
        gap: 12px;
        padding: 8px 0 24px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 32px;
    }
    .fv-logo {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.25em;
        color: var(--accent);
        text-transform: uppercase;
    }
    .fv-title {
        font-size: 28px;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.04em;
    }
    .fv-sub {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: var(--muted);
        letter-spacing: 0.15em;
        margin-left: auto;
    }

    /* Control row */
    .fv-controls {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }

    /* Selectbox and button overrides */
    [data-testid="stSelectbox"] > div > div {
        background: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
        color: var(--text) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 13px !important;
    }
    [data-testid="stSelectbox"] label {
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: var(--label) !important;
    }

    /* Primary button */
    .stButton > button {
        background: var(--accent) !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        padding: 12px 28px !important;
        width: 100% !important;
        transition: opacity 0.15s !important;
        white-space: nowrap !important;
        padding: 12px 18px !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }
    .stButton > button:disabled {
        background: var(--border) !important;
        color: var(--muted) !important;
        cursor: not-allowed !important;
    }

    /* Stance card */
    .stance-card {
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 28px 32px;
        text-align: center;
        background: var(--surface);
        position: relative;
        overflow: hidden;
    }
    .stance-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .stance-bull::before { background: var(--bull); }
    .stance-bear::before { background: var(--bear); }
    .stance-neutral::before { background: var(--neutral); }

    .stance-label {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--label);
        margin-bottom: 8px;
    }
    .stance-value {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .stance-bull .stance-value  { color: var(--bull); }
    .stance-bear .stance-value  { color: var(--bear); }
    .stance-neutral .stance-value { color: var(--neutral); }

    /* Conflict badge */
    .conflict-badge {
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 1px;
        margin-top: 12px;
    }
    .conflict-low { background: #1a2e1a; color: var(--bull); border: 1px solid var(--bull); }
    .conflict-medium { background: #2e2a1a; color: #f5c842;    border: 1px solid #f5c842; }
    .conflict-high { background: #2e1a1a; color: var(--bear); border: 1px solid var(--bear); }

    /* Confidence bar */
    .conf-wrap {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 20px 24px;
    }
    .conf-label {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--label);
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }
    .conf-track {
        height: 6px;
        background: var(--border);
        border-radius: 1px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 1px;
        transition: width 0.6s ease;
    }
    
    /* Info icon + hover tooltip, used in conf-label rows */
    .info-icon {
        position: relative;
        display: inline-block;
        cursor: help;
        color: var(--muted);
        font-size: 12px;
        margin-left: 4px;
    }
    .info-icon .tooltip-text {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        z-index: 10;
        bottom: 130%;
        left: 50%;
        transform: translateX(-50%);
        width: 220px;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 10px 12px;
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        line-height: 1.6;
        letter-spacing: 0.02em;
        text-transform: none;
        color: var(--text);
        transition: opacity 0.15s ease;
    }
    .info-icon:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /* Modality signal cards */
    .mod-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 16px 20px;
        height: 100%;
    }
    .mod-name {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--label);
        margin-bottom: 8px;
    }
    .mod-signal {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .mod-conf {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: var(--label);
        margin-bottom: 8px;
    }
    .mod-desc {
        font-size: 12px;
        color: var(--text);
        line-height: 1.5;
        opacity: 0.8;
    }
    .sig-bull  { color: var(--bull); }
    .sig-bear  { color: var(--bear); }
    .sig-neut  { color: var(--neutral); }

    /* Reasoning block */
    .reasoning-block {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 2px;
        padding: 20px 24px;
        font-size: 14px;
        line-height: 1.7;
        color: var(--text);
        opacity: 0.9;
    }

    /* Section headers */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--label);
        font-weight: 700;
        margin: 28px 0 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }
    .feedback-note {
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        color: var(--label);
        letter-spacing: 0.1em;
        margin-top: 8px;
    }

    /* History table */
    .hist-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px 14px;
        padding: 12px 0;
        border-bottom: 1px solid var(--border);
        font-size: 13px;
    }
    .hist-row:last-child { border-bottom: none; }
    .hist-tf {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: var(--label);
        min-width: 30px;
    }
    .hist-stance {
        font-weight: 700;
        font-size: 12px;
        min-width: 70px;
    }
    .hist-conf {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: var(--label);
        min-width: 55px;
    }
    .hist-conflict {
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        min-width: 60px;
    }
    .hist-time {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: var(--label);
        margin-left: auto;
    }
    .hist-outcome {
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        padding: 2px 6px;
        border-radius: 1px;
    }
    .outcome-correct { background: #1a2e1a; color: var(--bull); }
    .outcome-incorrect { background: #2e1a1a; color: var(--bear); }
    .outcome-pending { background: #26263a; color: var(--label); }

    /* Shrink the per-row history review button, overriding the global
        .stButton style which is sized for RUN ANALYSIS / SUBMIT OUTCOME */
    [class*="st-key-hist_btn_wrap_"] .stButton > button {
        background: transparent !important;
        color: var(--label) !important;
        border: none !important;
        padding: 2px 6px !important;
        font-size: 12px !important;
        width: auto !important;
        min-width: 0 !important;
        line-height: 1 !important;
    }
    [class*="st-key-hist_btn_wrap_"] .stButton > button:hover {
        color: var(--accent) !important;
        opacity: 1 !important;
    }
    /* Slim down PREV/NEXT pagination buttons */
    [class*="st-key-hist_page_btn_wrap"] .stButton > button {
        padding: 4px 10px !important;
        min-height: 12px !important;
        font-size: 8px !important;
        width: auto !important;
    }
    
    /* Empty state (inline, eg. "no history yet" row),
       this is not the right-column placeholder, see .output-panel below */
    .empty-state {
        text-align: center;
        padding: 40px 0;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: var(--muted);
        letter-spacing: 0.1em;
    }

    /* Right-column output panel, to wraps the pre-run placeholder, the
       loading state AND is the visual anchor the fusion output sits in.
       Gives the whole right side a clear 'this is a real panel' border so
       doesnt feel like deadspace/poor UI before a run. */
    .output-panel {
        border: 1px solid var(--border);
        border-radius: 2px;
        background: var(--surface);
        min-height: 420px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px;
    }
    .output-panel-icon {
        font-size: 48px;
        margin-bottom: 20px;
        opacity: 0.4;
        color: var(--accent);
    }
    .output-panel-text {
        font-family: 'Space Mono', monospace;
        font-size: 16px;
        letter-spacing: 0.15em;
        color: var(--label);
        line-height: 1.8;
    }
    .output-panel-icon.spinning {
        animation: fv-spin 1.4s linear infinite;
    }
    @keyframes fv-spin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }

    /* Metadata strip */
    .meta-strip {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: var(--label);
        letter-spacing: 0.12em;
        padding: 8px 0;
        border-top: 1px solid var(--border);
        margin-top: 32px;
    }

    /* Streamlit textarea/input */
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 2px !important;
        color: var(--text) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 12px !important;
    }
    [data-testid="stTextArea"] label,
    [data-testid="stTextInput"] label {
        font-family: 'Space Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
    }

    /* Radio buttons for outcome */
    [data-testid="stRadio"] label {
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        color: var(--text) !important;
    }
    [data-testid="stRadio"] > div > label {
        font-size: 11px !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: var(--label) !important;
    }

    /* Weight panel */
    .weight-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 20px 24px;
        margin-top: 12px;
    }
    .weight-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .weight-row:last-child { margin-bottom: 0; }
    .weight-name {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--label);
        min-width: 90px;
    }
    .weight-bar-track {
        flex: 1;
        height: 4px;
        background: var(--border);
        border-radius: 1px;
        overflow: hidden;
    }
    .weight-bar-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 1px;
        transition: width 0.5s ease;
    }
    .weight-pct {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: var(--text);
        min-width: 32px;
        text-align: right;
    }
    .weight-mult {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: var(--label);
        min-width: 44px;
        text-align: right;
    }

    div[data-baseweb="notification"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)