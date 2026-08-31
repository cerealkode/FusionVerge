import streamlit as st
import json
import os
import psycopg2
import psycopg2.extras

@st.cache_resource
def get_db_connection():
    """Opens a single persistent PostgreSQL connection for the application session.

    - Cached with st.cache_resource so it is not reopened on every rerender
    - Hardstop immediately if env config missing so no downstream crashes
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        st.error("DATABASE_URL not set in .env. PostgreSQL connection required.")
        st.stop()
    conn = psycopg2.connect(url)
    # Ensure all executions commit instantly
    # *avoid uncommitted transaction blocks locking the tables
    conn.autocommit = True
    return conn

def init_db(conn):
    """Creates feedback_log and weight_log tables if they do not exist.
    
    - Called once at startup
    """
    with conn.cursor() as cur:
        # Create feedback_log table to capture full payload of every pipeline run
        # * JSONB fields (signals, modality_outputs) allows for fast index and query on nested data structs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback_log (
                id                SERIAL PRIMARY KEY,
                pair              VARCHAR(10)  NOT NULL,
                timeframe         VARCHAR(5)   NOT NULL,
                run_date          DATE         NOT NULL,
                run_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                stance            VARCHAR(10)  NOT NULL,
                confidence        FLOAT        NOT NULL,
                conflict_level    VARCHAR(10)  NOT NULL,
                conflict_imbalance  FLOAT,
                signals           JSONB        NOT NULL,
                reasoning         TEXT         NOT NULL,
                modality_outputs  JSONB        NOT NULL,
                outcome           VARCHAR(15)  NULL
            );
        """)
        # Create weight_log table to track historical record of weight adjustments
        # * FK reference feedback_log as performance metrics is tied to weight adjustment
        cur.execute("""
            CREATE TABLE IF NOT EXISTS weight_log (
                id           SERIAL PRIMARY KEY,
                feedback_id  INTEGER      REFERENCES feedback_log(id),
                logged_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                weights      JSONB        NOT NULL
            );
        """)
        # Inject composite index on pair and run_at timestamp
        # * best practice for fast querying of fetch_history later. eg. WHERE pair = ? ORDER BY run_at DESC
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_pair
            ON feedback_log(pair, run_at DESC);
        """)

        # Add conflict_imbalance to feedback_log (iteration 2 migration)
        # * so conflict imbalance can be recovered for historical runs
        # * pre-migration rows will have NULL
        cur.execute("""
            ALTER TABLE feedback_log
            ADD COLUMN IF NOT EXISTS conflict_imbalance FLOAT;
        """)

def insert_analysis(conn, result: dict) -> int:
    """Persists a completed analysis run to feedback_log.

    - Uses json.dumps to convert nested modality payloads into strings before saving
    - Returns the new row id so the session can reference it for feedback forms later
    - modality_outputs stored as JSONB for Stage 4 weight update queries
    
    Returns the auto-generated row ID (integer).
    """
    with conn.cursor() as cur:
        # Insert analysis results into feedback_log table
        # * uses parameterized SQL (%s) to safely pass values into the query
        # * RETURNING id lets us grab the new row ID immediately without running an extra SELECT query
        cur.execute("""
            INSERT INTO feedback_log
                (pair, timeframe, run_date, stance, confidence, conflict_level,
                 conflict_imbalance, signals, reasoning, modality_outputs)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            result["pair"],
            result["timeframe"],
            result["date"],
            result["stance"],
            result["confidence"],
            result["conflict_level"],
            result.get("conflict_imbalance"), # NULL if not present, kept optional on purpose (premigration row is null)
            json.dumps(result["signals"]),
            result["reasoning"],
            json.dumps(result["modality_outputs"]),
        ))
        return cur.fetchone()[0]

def update_outcome(conn, row_id: int, outcome: str):
    """Updates the outcome and notes for a completed analysis.

    - Called when the user submits feedback from the history panel (on dashboard)
    - Outcome options: 'correct', 'incorrect', 'uncertain'
    - Feedback loop logic on adding weight update will trigger here
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE feedback_log
            SET outcome = %s
            WHERE id = %s
        """, (outcome, row_id))

def fetch_analysis_by_id(conn, row_id: int) -> dict | None:
    """Fetches a single full feedback_log row by id.
 
    - Used to reconstruct the right-hand fusion output panel for a past run when
    - Unlike fetch_history() this pulls every column needed to redraw the panel
 
    Returns a dict of the row, or None if no row with that id exists.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, pair, timeframe, run_date, stance, confidence,
                   conflict_level, conflict_imbalance, signals, reasoning,
                   modality_outputs, outcome
            FROM feedback_log
            WHERE id = %s
        """, (row_id,))
        return cur.fetchone()

def fetch_history(conn, pair: str, limit: int = 5, offset: int = 0) -> list:
    """Returns the analyses for the given pair, lastest to oldest.

    - offset=0 is page 1, =1 is page 2, etc

    Returns list of dictionaries containing individual log row keys and data values.
    """
    # Use RealDictCursor to return database rows as clean python dictionaries
    # * https://www.psycopg.org/docs/extras.html
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, pair, timeframe, run_at, stance, confidence,
                   conflict_level, reasoning, outcome
            FROM feedback_log
            WHERE pair = %s
            ORDER BY run_at DESC
            LIMIT %s OFFSET %s
        """, (pair, limit, offset))
        return cur.fetchall()
    
def fetch_history_count(conn, pair: str) -> int:
    """Returns the total number of logged analyses for the given pair.

    - Paired with fetch_history()'s offset to compute total pages for the pagination controls in ui.views.render_history_panel()
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM feedback_log WHERE pair = %s", (pair,))
        return cur.fetchone()[0]