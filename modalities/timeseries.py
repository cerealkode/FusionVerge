import os
import json
import logging
import joblib
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from tensorflow.keras.models import Model, load_model
from infrastructure.config import SUPPORTED_ASSETS, TIMEFRAME_CONFIGS, OHLC, OHLC_ALL, GRU_BEST_CONFIGS_PATH, GRU_SAVEPATH_TEMPLATE, GRU_SCALER_TEMPLATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_config(pair: str, timeframe: str) -> dict:
    """Reads best_configs.json (produced by the feature prototype training notebook)

    - Reads the winning feature set and lookback from best_configs.json (saved model)
    - Hard fails if the config file or requested entry is missing
      (we dont want it to retrain from scratch when deployed/hosted)
    
    Returns config dict.
    """
    if not os.path.exists(GRU_BEST_CONFIGS_PATH):
        raise FileNotFoundError(
            f"best_configs.json not found at {GRU_BEST_CONFIGS_PATH}. "
            f"Run the training notebook to generate it."
        )

    with open(GRU_BEST_CONFIGS_PATH, "r") as f:
        best_configs = json.load(f)

    key = f"{pair}_{timeframe}" # config key format eg. EURUSD_1h
    if key not in best_configs:
        raise KeyError(
            f"No entry for '{key}' in {GRU_BEST_CONFIGS_PATH}. "
            f"Check that training completed for this pair and timeframe."
        )

    cfg = best_configs[key]
    logger.info(f"Best config for {key}: features={cfg['features']} lookback={cfg['lookback']} acc={cfg.get('accuracy')}%")
    return cfg

def fetch_ohlcv(pair: str, timeframe: str) -> pd.DataFrame:
    """Fetches OHLCV data for the given pair and timeframe from yfinance.

    - Very similar to chart.py's fetch_ohlcv(), function is not shared to keep each pipeline self-contained
    - Uses TIMEFRAME_CONFIGS to match the training data settings

    Returns a cleaned DataFrame with OHLC columns (no Volume as not used in training).
    """
    ticker = SUPPORTED_ASSETS.get(pair)
    if not ticker:
        raise ValueError(f"Unsupported pair: {pair}. Check SUPPORTED_ASSETS in config.")

    tf_cfg = TIMEFRAME_CONFIGS.get(timeframe)
    if not tf_cfg:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Check TIMEFRAME_CONFIGS in config.")

    period = tf_cfg["period"]
    interval = tf_cfg["interval"]
    logger.info(f"Fetching OHLCV for {pair} ({ticker}) | interval={interval} period={period}...")

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}.")

    # Flatten columns if yfinance returns a MultiIndex
    # * Source - https://stackoverflow.com/a/21081062, Posted by jonrsharpe, Retrieved 2026-07-06, License - CC BY-SA 4.0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].dropna() # keep only OHLC columns used in training
    logger.info(f"Fetched {len(df)} rows for {pair} {timeframe}.")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators used during GRU training.
    
    - Using pandas-ta, matching the training feature prototype notebook exactly for feature consistency
    - RSI(14), MACD line (12/26/9), EMA20, ATR(14)
    
    Returns DataFrame with indicators added.
    """
    df = df.copy()
    
    # RSI(14)
    df["RSI"] = ta.rsi(df["Close"], length=14)
    # MACD line (12/26/9)
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    # EMA(20)
    df["EMA20"] = ta.ema(df["Close"], length=20)
    # ATR(14) - volatitlity
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    df.dropna(inplace=True) # remove indicator warmup rows
    logger.info(f"Indicators computed. Rows after dropna: {len(df)}")
    return df

def resolve_features(feature_list: list) -> list:
    """ Validate feature list loaded from best_config.json.

    - Maps feature list stored in best_configs.json back to the actual column name lists defined in config

    Returns validated feature list (eg. ["Open","High","Low","Close"]).
    """
    valid = set(OHLC_ALL)
    # Ensure every requested feature exist in supported feature set
    unknown = [f for f in feature_list if f not in valid]
    if unknown:
        raise ValueError(
            f"best_configs.json contains unrecognised features: {unknown}. "
            f"Valid features are: {OHLC_ALL}"
        )
    return feature_list

def load_gru(pair: str, timeframe: str) -> tuple[Model, object]:
    """Loads the pre-trained GRU model and its fitted scaler from the models/ directory

    - Scaler was saved with joblib in the training jupyter notebook (in feature prototype).
    - Hard fails if either file is missing, computationally too expensive for silent retrain.
    
    Return (model, scaler).
    """
    model_path = GRU_SAVEPATH_TEMPLATE.format(pair=pair, timeframe=timeframe)
    scaler_path = GRU_SCALER_TEMPLATE.format(pair=pair, timeframe=timeframe)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model at {model_path}. "
            f"Run the training notebook for {pair} {timeframe} first."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No scaler at {scaler_path}. "
            f"Run the training notebook for {pair} {timeframe} first."
        )

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    logger.info(f"Loaded model and scaler for {pair} {timeframe}.")
    return model, scaler

def run_gru_inference(model: Model, scaler, df: pd.DataFrame, features: list, lookback: int) -> dict:
    """Runs GRU inference on the most recent `lookback` rows of fresh data.
    
    - Feature selection and ordering use the list from best_configs.json to guarantee
      exact alignment with how the scaler and model were trained
    - Scales with the pre-fitted scaler so no refit on inference data.

    Returns dict with keys: signal (+1 or -1), confidence, and direction_probability.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing features required by model: {missing}")

    data = df[features].values
    # Scale using the training scaler (never refit on inference data)
    data_scaled = scaler.transform(data)

    if len(data_scaled) < lookback:
        raise ValueError(
            f"Not enough rows after indicator warmup: have {len(data_scaled)}, need {lookback}. "
            f"Try a longer fetch period or check indicator computation."
        )

    X = data_scaled[-lookback:].reshape(1, lookback, len(features)) # use latest lookback window
    preds = model.predict(X, verbose=0)
    dir_prob = float(preds[1].flatten()[0])

    # 0.5 porbability threshold for binary direction prediction
    signal = 1 if dir_prob > 0.5 else -1
    confidence = round(dir_prob if dir_prob > 0.5 else 1 - dir_prob, 4)

    logger.info(f"GRU inference: dir_prob={dir_prob:.4f} signal={signal:+d} conf={confidence:.4f}")
    return {
        "signal": signal,
        "confidence": confidence,
        "direction_probability": round(dir_prob, 4)
    }

def get_timeseries_signal(pair: str, timeframe: str) -> dict:
    """Main entry point for the time-series modality.

    - Read and load best config, model, scaler
    - Fetch fresh OHLC data, compute indicators, then run GRU inference

    Returns dict with keys: signal, confidence, direction_probability.
    """
    cfg = load_best_config(pair, timeframe)
    features = resolve_features(cfg["features"])
    lookback = cfg["lookback"]

    model, scaler = load_gru(pair, timeframe)

    df = fetch_ohlcv(pair, timeframe)

    # Only compute indicators if the winning config (in best models) uses OHLC_ALL
    # * if it only used OHLC, indicators are not needed so we skip the pandas-ta step
    needs_indicators = any(f not in OHLC for f in features)
    if needs_indicators:
        df = compute_indicators(df)

    result = run_gru_inference(model, scaler, df, features, lookback)
    logger.info(f"Time-series signal for {pair} {timeframe}: {result}")
    return result


# Run smoke test with python -m modalities.timeseries
if __name__ == "__main__":
    result = get_timeseries_signal("EURUSD", "1h")
    print(result)