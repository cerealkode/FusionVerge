import os
import uuid
import logging
import yfinance as yf
import matplotlib # matplotlib needs import before mpl, else without the headless it will keep crashing during the test
matplotlib.use("Agg") # headless backend
import mplfinance as mpf
import pandas as pd
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import copy
from infrastructure.config import SUPPORTED_ASSETS, CHART_HISTORY, CHART_STYLE, CHART_DPI, CHART_OUTPUT_DIR, YOLO_MODEL_REPO, YOLO_MODEL_FILE, YOLO_CLASSES, YOLO_SIGNAL_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_ohlcv(pair: str, timeframe: str) -> pd.DataFrame:
    """Fetches OHLC data for a pair and timeframe from yfinance.

    - Volume excluded as not used for pattern detection

    Returns a cleaned DataFrame with OHLC columns only.
    """
    ticker = SUPPORTED_ASSETS.get(pair)
    if not ticker:
        raise ValueError(f"Unsupported pair: {pair}. Check SUPPORTED_ASSETS in config.")

    history = CHART_HISTORY.get(timeframe)
    if not history:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Check CHART_HISTORY in config.")

    logger.info(f"Fetching OHLC for {pair} ({ticker}) at {timeframe} over {history}...")
    df = yf.download(ticker, period=history, interval=timeframe, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}. Check ticker or network.")

    # Flatten columns if yfinance returns MultiIndex (edgecase)
    # * Source - https://stackoverflow.com/a/21081062, Posted by jonrsharpe, Retrieved 2026-07-06, License - CC BY-SA 4.0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index.name = "Date"
    logger.info(f"Fetched {len(df)} rows for {pair}.")
    return df

def generate_chart(df: pd.DataFrame, output_dir: str = CHART_OUTPUT_DIR) -> str:
    """Generate and save candlestick chart from OHLC data.

    - Chart required for the YOLO model to run inference on

    Returns the file path of the saved image.
    """

    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns] # validates required col before plot
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    os.makedirs(output_dir, exist_ok=True) # ensure output folder exists

    # Unique filename per call so concurrent runs donnt collide/overwrite each other
    savepath = os.path.join(output_dir, f"chart_{uuid.uuid4().hex[:8]}.png")

    logger.info(f"Generating chart image to {savepath}...")
    # Carried over from feature prototype
    mpf.plot(
        df,
        type="candle",
        style=CHART_STYLE,
        volume=False,
        savefig=dict(fname=savepath, dpi=CHART_DPI, bbox_inches="tight")
    )
    logger.info("Chart image saved.")
    return savepath

def load_yolo_model() -> YOLO:
    """Load pretrained YOLO chart detection model

    - source and usage: https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8 and https://github.com/foduucom/Stockmarket-pattern-detection

    Downloads the model on first use and reuse local cache afterwards.
    """
    logger.info(f"Loading YOLO model from {YOLO_MODEL_REPO}...")
    model_path = hf_hub_download(
        repo_id=YOLO_MODEL_REPO,
        filename=YOLO_MODEL_FILE
    )
    logger.info("YOLO model loaded.")
    return YOLO(model_path)

def run_yolo_inference(model: YOLO, chart_path: str, save_annotated: bool = False) -> dict:
    """Runs YOLO inference on chart image.

    - Uses the highest-confidence detected pattern if multiple patterns are found
    - Returns a neutral signal if no pattern is detected
    - Optionally have a save_annotated flag to show bounding boxes of identified pattern, used in testing only

    Returns dict with keys: signal, confidence, pattern.
    """
    results = model(chart_path, verbose=False)
    boxes = results[0].boxes

    if boxes and len(boxes) > 0:
        # Take highest confidence detection score
        confidences = boxes.conf.tolist()
        best_idx = confidences.index(max(confidences))
        cls_idx = int(boxes.cls[best_idx].item())
        conf = confidences[best_idx]
        label = YOLO_CLASSES[cls_idx] if cls_idx < len(YOLO_CLASSES) else "none"
    else:
        logger.info("YOLO detected no patterns in chart.")
        label = "none"
        conf = 0.0

    # Convert detected pattern into a directional trading signal
    signal = YOLO_SIGNAL_MAP.get(label, 0)

    logger.info(f"YOLO result: {label} | signal={signal:+d} | conf={conf:.3f}")

    # Save annotated chart showing only the highest-confidence detection
    if save_annotated and boxes and len(boxes) > 0:
        best_result = copy.deepcopy(results[0])
        best_result.boxes = best_result.boxes[best_idx:best_idx + 1]
        best_result.save(filename="chart_output_annotated.png")
    return {
        "signal": signal,
        "confidence": round(conf, 4),
        "pattern": label
    }

def get_chart_signal(pair: str, timeframe: str, yolo_model: YOLO = None, save_annotated: bool = False) -> dict:
    """Main entry point for the chart modality.

    - Runs full pipeline: fetch OHLC -> generate chart -> YOLO inference
    - Accepts optional preloaded YOLO model to avoid repeated model loads

    Returns dict with keys: signal, confidence, pattern.
    """
    if yolo_model is None:
        yolo_model = load_yolo_model()

    df = fetch_ohlcv(pair, timeframe)
    chart_path = generate_chart(df)
    result = run_yolo_inference(yolo_model, chart_path, save_annotated=save_annotated)

    logger.info(f"Chart signal for {pair} {timeframe}: {result}")
    return result


# Run smoke test with python -m modalities.chart
if __name__ == "__main__":
    result = get_chart_signal("EURUSD", "1h", save_annotated=True)
    print(result)