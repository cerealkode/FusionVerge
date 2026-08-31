import logging
from datetime import date
from modalities.sentiment import load_sentiment_model, get_sentiment_signal
from modalities.chart import load_yolo_model, get_chart_signal
from modalities.timeseries import get_timeseries_signal
from modalities.positioning import get_positioning_signal
from core.fusion import get_fusion_output
from core.feedback import load_weights, apply_weights
from infrastructure.config import SUPPORTED_ASSETS, TIMEFRAME_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates the full multi-modal end-to-end analysis pipeline.

    - Loads heavy models once (FinBERT, YOLO) at __init__,
      to avoid repeated load cost on every analysis
      (~30-60s FinBERT reload and big downloads from YOLO model)
    - Runs all 4 modalities: news, chart, timeseries, positioning
    - Applies per-pair weights and send outputs into fusion layer (LLM)
    
    Returns final structured result with metadata.
    """

    def __init__(self, db_conn=None):
        """Init orchestrator and preload models.

        - Models loaded only once
        - Raises and fail immediately if either model cant load (instead of mid-analysis)
        - Optional DB connection for loading weight history. Defaults to 1.0 if None
        """
        logger.info("Orchestrator initialising - loading FinBERT...")
        self.sentiment_pipe = load_sentiment_model()
        logger.info("FinBERT loaded.")

        logger.info("Orchestrator initialising - loading YOLO...")
        self.yolo_model = load_yolo_model()
        logger.info("YOLO loaded.")

        # Store DB connection for the feedback loop weights
        self.db_conn = db_conn
        logger.info("Orchestrator ready.")

    def run(self, pair: str, timeframe: str) -> dict:
        """Runs the full analysis pipeline for the given pair and timeframe.

        Steps:
            1. Validate pair and timeframe against config
            2. Run all four modality pipelines in sequence
            3. Load per-pair weights from DB (defaults to 1.0 if no history)
            4. Apply weight multipliers to modality confidence scores
            5. Pass weighted outputs to fusion layer for conflict scoring,
               LLM reasoning and structured output
            6. Attach metadata and return

        Returns structured output dict with keys:
            pair, timeframe, date, stance, confidence, conflict_level,
            signals, reasoning, modality_outputs, weights
            - modality_outputs contains effective (weighted) confidence values
            - weights contains the raw multiplier dict used for this run
        """
        # Validate inputs first
        if pair not in SUPPORTED_ASSETS:
            raise ValueError(
                f"Unsupported pair: {pair}. "
                f"Supported pairs: {list(SUPPORTED_ASSETS.keys())}"
            )
        if timeframe not in TIMEFRAME_CONFIGS:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported timeframes: {list(TIMEFRAME_CONFIGS.keys())}"
            )

        logger.info(f"Running analysis for {pair} {timeframe}...")

        # Modality 1: Sentiment (news)
        logger.info(f"[1/4] Sentiment...")
        try:
            news_output = get_sentiment_signal(pair, pipe=self.sentiment_pipe)
        except Exception as e:
            # Fallback to neutral so as to not break pipeline
            logger.error(f"Sentiment modality failed: {e}. Using neutral fallback.")
            news_output = {"signal": 0, "confidence": 0.0, "distribution": {"positive": 0, "negative": 0, "neutral": 0}}

        # Modality 2: Chart (YOLO pattern detection)
        logger.info(f"[2/4] Chart...")
        try:
            chart_output = get_chart_signal(pair, timeframe, yolo_model=self.yolo_model)
        except Exception as e:
            # Fallback to no pattern if chart fails, so as to not break pipeline
            logger.error(f"Chart modality failed: {e}. Using neutral fallback.")
            chart_output = {"signal": 0, "confidence": 0.0, "pattern": "none"}

        # Modality 3: Time-series (custom trained GRU)
        logger.info(f"[3/4] Time-series...")
        try:
            ts_output = get_timeseries_signal(pair, timeframe)
        except Exception as e:
            # Fallback to neutral so as to not break pipeline
            logger.error(f"Time-series modality failed: {e}. Using neutral fallback.")
            ts_output = {"signal": 0, "confidence": 0.0, "direction_probability": 0.5}

        # Modality 4: Positioning (myfxbook API for contrarian rule)
        logger.info(f"[4/4] Positioning...")
        try:
            pos_output = get_positioning_signal(pair)
        except Exception as e:
            # External API failure fallback
            # * possible reasoning is too much activity in short span of time causing timeouts
            logger.error(f"Positioning modality failed: {e}. Using neutral fallback.")
            pos_output = {"signal": 0, "confidence": 0.0, "long_pct": None, "short_pct": None, "note": "fallback"}

        # Consolidate raw modality outputs
        modality_outputs = {
            "news": news_output,
            "chart": chart_output,
            "timeseries": ts_output,
            "positioning": pos_output,
        }
        
        # Apply weights to confidence BEFORE fusion layer
        # * per-pair weights with applied confidence multipliers
        # * default to 1.0 weight if no DB connection/history, as mentioned above
        # * more details in feedback.py and fusion.py (for application)
        if self.db_conn is not None:
            weights = load_weights(self.db_conn, pair)
        else:
            from infrastructure.config import DEFAULT_WEIGHT
            weights = {m: DEFAULT_WEIGHT for m in modality_outputs}
            logger.info("No DB connection - using default weights (1.0).")

        weighted_outputs = apply_weights(modality_outputs, weights)
        logger.info(f"Weights applied for {pair}: {weights}")

        logger.info("All modalities complete. Running fusion...")

        # Fusion layer (receives weighted confidence values)
        fusion_result = get_fusion_output(pair, timeframe, weighted_outputs)

        # Attach metadata
        # * for downstream use (dashboard/logging/feedback loop)
        result = {
            "pair": pair,
            "timeframe": timeframe,
            "date": str(date.today()),
            **fusion_result,
            "modality_outputs": weighted_outputs, # weighted outputs (effective confidence used in decision)
            "weights": weights, # raw multiplier dict for transparency/dashboard display
        }

        logger.info(
            f"Analysis complete - {pair} {timeframe}: "
            f"stance={result.get('stance')} "
            f"confidence={result.get('confidence')} "
            f"conflict={result.get('conflict_level')}"
        )
        return result