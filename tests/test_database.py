"""
Unit tests for the database layer (infrastructure/database.py)
Uses a real PostgreSQL connection (TEST_DATABASE_URL or DATABASE_URL) - tests are skipped gracefully if no reachable database is configured.
Test rows are tagged with a [TEST] marker in reasoning and cleaned up after the module runs.

Usage - python -m pytest tests/test_database.py -v --disable-warnings
"""
import os
from dotenv import load_dotenv
load_dotenv() # need this for the database url
import pytest
import psycopg2
import psycopg2.extras
from datetime import date
from infrastructure.database import init_db, insert_analysis, update_outcome, fetch_history


# PSQL availability check
def db_reachable() -> bool:
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        return False
    try:
        conn = psycopg2.connect(url)
        conn.close()
        return True
    except Exception:
        return False

requires_db = pytest.mark.skipif(
    not db_reachable(),
    reason="PostgreSQL not reachable. Set TEST_DATABASE_URL or DATABASE_URL in .env."
)


# DB fixtures
@pytest.fixture(scope="module")
def db_conn():
    """Shared database connection for the test module.
    
    - use a [TEST] prefix in reasoning so rows are identifiable, removes it after done
    """
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    conn = psycopg2.connect(url)
    conn.autocommit = True
    init_db(conn)
    yield conn
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM feedback_log WHERE reasoning LIKE %s",
            ("%[TEST]%",)
        )
    conn.close()

@pytest.fixture
def sample_result():
    """Minimal mock pipeline result dict matching structure from orchestrator.run()
    
    - used by insert_analysis() test
    """
    return {
        "pair": "EURUSD",
        "timeframe": "1h",
        "date": str(date.today()),
        "stance": "bearish",
        "confidence": 0.62,
        "conflict_level": "medium",
        "signals": {
            "news": "[TEST] Negative EUR sentiment dominates.",
            "chart": "[TEST] Head and shoulders top detected.",
            "timeseries": "[TEST] GRU predicts downward movement.",
            "positioning": "[TEST] Retail 65% long, contrarian bearish.",
        },
        "reasoning": "[TEST] Three bearish modalities outweigh one neutral.",
        "modality_outputs": {
            "news": {"signal": -1, "confidence": 0.79, "raw_confidence": 0.79,
                     "weight": 1.0, "distribution": {"positive": 3, "negative": 8, "neutral": 1}},
            "chart": {"signal": -1, "confidence": 0.55, "raw_confidence": 0.55,
                      "weight": 1.0, "pattern": "Head and shoulders top"},
            "timeseries": {"signal": -1, "confidence": 0.62, "raw_confidence": 0.62,
                           "weight": 1.0, "direction_probability": 0.38},
            "positioning": {"signal":  0, "confidence": 0.50, "raw_confidence": 0.50,
                            "weight": 1.0, "long_pct": 65, "short_pct": 35, "note": "fallback"},
        },
        "weights": {
            "news": 1.0, "chart": 1.0, "timeseries": 1.0, "positioning": 1.0
        },
    }


# init_db function testing
@requires_db
class TestInitDb:
    def test_creates_feedback_log_table(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feedback_log")
            count = cur.fetchone()[0]
        assert isinstance(count, int)

    def test_creates_weight_log_table(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM weight_log")
            count = cur.fetchone()[0]
        assert isinstance(count, int)

    def test_idempotent_on_repeated_calls(self, db_conn):
        init_db(db_conn)
        init_db(db_conn)
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feedback_log")
            assert cur.fetchone()[0] >= 0

    def test_feedback_log_has_required_columns(self, db_conn):
        required = {
            "id", "pair", "timeframe", "run_date", "run_at",
            "stance", "confidence", "conflict_level",
            "signals", "reasoning", "modality_outputs",
            "outcome", "outcome_notes",
        }
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'feedback_log'
            """)
            columns = {row[0] for row in cur.fetchall()}
        missing = required - columns
        assert not missing, f"feedback_log missing columns: {missing}"

    def test_weight_log_has_required_columns(self, db_conn):
        required = {"id", "feedback_id", "logged_at", "weights"}
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'weight_log'
            """)
            columns = {row[0] for row in cur.fetchall()}
        missing = required - columns
        assert not missing, f"weight_log missing columns: {missing}"


# insert_analysis() function testing
@requires_db
class TestInsertAnalysis:
    def test_returns_integer_id(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_stored_values_match_input(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM feedback_log WHERE id = %s", (row_id,))
            row = cur.fetchone()
        assert row["pair"] == sample_result["pair"]
        assert row["timeframe"] == sample_result["timeframe"]
        assert row["stance"] == sample_result["stance"]
        assert abs(row["confidence"] - sample_result["confidence"]) < 1e-6
        assert row["conflict_level"] == sample_result["conflict_level"]
        assert row["reasoning"] == sample_result["reasoning"]

    def test_signals_stored_as_jsonb(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT signals FROM feedback_log WHERE id = %s", (row_id,))
            row = cur.fetchone()
        signals = row["signals"]
        for key in ["news", "chart", "timeseries", "positioning"]:
            assert key in signals

    def test_modality_outputs_stored_as_jsonb(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT modality_outputs FROM feedback_log WHERE id = %s", (row_id,)
            )
            row = cur.fetchone()
        outputs = row["modality_outputs"]
        for key in ["news", "chart", "timeseries", "positioning"]:
            assert key in outputs

    def test_outcome_is_null_on_insert(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        with db_conn.cursor() as cur:
            cur.execute("SELECT outcome FROM feedback_log WHERE id = %s", (row_id,))
            outcome = cur.fetchone()[0]
        assert outcome is None

    def test_multiple_inserts_get_unique_ids(self, db_conn, sample_result):
        id1 = insert_analysis(db_conn, sample_result)
        id2 = insert_analysis(db_conn, sample_result)
        assert id1 != id2


# update_outcome() function testing
@requires_db
class TestUpdateOutcome:
    def test_outcome_written_correctly(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        update_outcome(db_conn, row_id, "correct")
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT outcome FROM feedback_log WHERE id = %s",
                (row_id,)
            )
            outcome = cur.fetchone()[0]
        assert outcome == "correct"

    def test_all_valid_outcome_values_accepted(self, db_conn, sample_result):
        for value in ["correct", "incorrect", "uncertain"]:
            row_id = insert_analysis(db_conn, sample_result)
            update_outcome(db_conn, row_id, value)
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT outcome FROM feedback_log WHERE id = %s", (row_id,)
                )
                assert cur.fetchone()[0] == value

    def test_outcome_can_be_updated_twice(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        update_outcome(db_conn, row_id, "correct")
        update_outcome(db_conn, row_id, "incorrect")
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT outcome FROM feedback_log WHERE id = %s",
                (row_id,)
            )
            outcome = cur.fetchone()[0]
        assert outcome == "incorrect"


# fetch_history() function testing
@requires_db
class TestFetchHistory:
    def test_returns_only_requested_pair(self, db_conn, sample_result):
        gbp_result = {**sample_result, "pair": "GBPUSD"}
        insert_analysis(db_conn, sample_result)
        insert_analysis(db_conn, gbp_result)
        rows = fetch_history(db_conn, "EURUSD", limit=10)
        for row in rows:
            assert row["pair"] == "EURUSD"

    def test_limit_respected(self, db_conn, sample_result):
        for _ in range(7):
            insert_analysis(db_conn, sample_result)
        rows = fetch_history(db_conn, "EURUSD", limit=5)
        assert len(rows) <= 5

    def test_ordered_newest_first(self, db_conn, sample_result):
        id1 = insert_analysis(db_conn, sample_result)
        id2 = insert_analysis(db_conn, sample_result)
        rows = fetch_history(db_conn, "EURUSD", limit=5)
        ids = [r["id"] for r in rows]
        assert ids.index(id2) < ids.index(id1)

    def test_returns_empty_list_for_unknown_pair(self, db_conn):
        rows = fetch_history(db_conn, "NZDUSD", limit=5)
        assert rows == []

    def test_required_keys_present_in_each_row(self, db_conn, sample_result):
        insert_analysis(db_conn, sample_result)
        rows = fetch_history(db_conn, "EURUSD", limit=1)
        assert rows
        required = {"id", "pair", "timeframe", "run_at", "stance",
                    "confidence", "conflict_level", "reasoning", "outcome"}
        for key in required:
            assert key in rows[0], f"Missing key in history row: {key}"

    def test_outcome_field_is_none_before_feedback(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        rows = fetch_history(db_conn, "EURUSD", limit=10)
        matching = [r for r in rows if r["id"] == row_id]
        assert matching
        assert matching[0]["outcome"] is None

    def test_outcome_visible_in_history_after_update(self, db_conn, sample_result):
        row_id = insert_analysis(db_conn, sample_result)
        update_outcome(db_conn, row_id, "correct")
        rows = fetch_history(db_conn, "EURUSD", limit=10)
        matching = [r for r in rows if r["id"] == row_id]
        assert matching
        assert matching[0]["outcome"] == "correct"