# To store constants and configurations for the pipeline

# Supported assets and their yfinance tickers
# Pairs where USD is quote use PAIR=X format
# Pairs where USD is base use the non-USD currency + =X
SUPPORTED_ASSETS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "USDCAD": "CAD=X",
}

# Timeframe configs - period/interval fed directly to yfinance
# Periods match training notebook to ensure indicator warmup rows are available
TIMEFRAME_CONFIGS = {
    "1h": {"period": "300d", "interval": "1h"},
    "1d": {"period": "10y", "interval": "1d"},
}

# Sentiment
SENTIMENT_MODEL = "ProsusAI/finbert"

# Two RSS feeds used for coverage and redundancy
RSS_FEEDS = [
    "https://www.fxstreet.com/rss/news",
    "https://www.forexlive.com/feed/news",
]

# Per-currency keywords used for both RSS headline filtering and currency focus detection.
# Replaces the old per-pair ASSET_KEYWORDS - no combined pair version needed.
# Known limitation: USD keywords (dollar, greenback, Fed) are high-frequency and will
# match headlines across many pairs. This is unavoidable without NLP entity resolution.
CURRENCY_KEYWORDS = {
    "EUR": ["EUR", "euro", "eurozone", "ECB", "European Central Bank"],
    "GBP": ["GBP", "pound", "sterling", "cable", "BOE", "Bank of England"],
    "AUD": ["AUD", "aussie", "RBA", "Reserve Bank of Australia"],
    "NZD": ["NZD", "kiwi", "RBNZ", "Reserve Bank of New Zealand"],
    "USD": ["USD", "dollar", "greenback", "Fed", "Federal Reserve", "FOMC"],
    "JPY": ["JPY", "yen", "BOJ", "Bank of Japan"],
    "CHF": ["CHF", "franc", "SNB", "Swiss National Bank"],
    "CAD": ["CAD", "loonie", "BOC", "Bank of Canada", "oil"],
}

# Directional effect of each currency on its pair when sentiment is positive
# base currency positive = bullish pair, quote currency positive = bearish pair
# neutral = 0 for headlines that mention the pair but without clear directional bias
PAIR_CURRENCY_EFFECT = {
    "EURUSD": {"EUR": 1, "USD": -1, "neutral": 0},
    "GBPUSD": {"GBP": 1, "USD": -1, "neutral": 0},
    "AUDUSD": {"AUD": 1, "USD": -1, "neutral": 0},
    "NZDUSD": {"NZD": 1, "USD": -1, "neutral": 0},
    "USDJPY": {"USD": 1, "JPY": -1, "neutral": 0},
    "USDCHF": {"USD": 1, "CHF": -1, "neutral": 0},
    "USDCAD": {"USD": 1, "CAD": -1, "neutral": 0},
}

SUPPORTED_PAIRS = { "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"}

# Chart
CHART_STYLE = "charles"
CHART_DPI = 150
CHART_HISTORY = {
    "1h": "14d",
    "1d": "120d"
}
CHART_OUTPUT_DIR = "chart_output"
YOLO_MODEL_REPO = "foduucom/stockmarket-pattern-detection-yolov8"
YOLO_MODEL_FILE = "model.pt"
YOLO_CLASSES = [
    "Head and shoulders bottom",
    "Head and shoulders top",
    "M_Head",
    "StockLine",
    "Triangle",
    "W_Bottom",
]
YOLO_SIGNAL_MAP = {
    "Head and shoulders top": -1,
    "Head and shoulders bottom": 1,
    "M_Head": -1,
    "W_Bottom": 1,
    "Triangle": 0,
    "StockLine": 0,
    "none": 0,
}

# Time-series
# Feature set definitions - the winning config per pair/timeframe is read from
# best_configs.json at runtime, so inference always uses the correct feature set
OHLC = ["Open", "High", "Low", "Close"]
OHLC_ALL = OHLC + ["RSI", "MACD", "EMA20", "ATR"]
# Path to best_configs.json produced by the training notebook
GRU_BEST_CONFIGS_PATH = "models/best_configs.json"
# Model and scaler path templates - must match filenames saved from Jupyter training
GRU_SAVEPATH_TEMPLATE = "models/{pair}_{timeframe}.keras"
GRU_SCALER_TEMPLATE = "models/{pair}_{timeframe}_scaler.pkl"

# Positioning
MYFXBOOK_URL = "https://www.myfxbook.com/api/get-community-outlook.json"
MYFXBOOK_LOGIN_URL = "https://www.myfxbook.com/api/login.json"
MYFXBOOK_LOGOUT_URL = "https://www.myfxbook.com/api/logout.json"
CONTRARIAN_THRESHOLD = 60

# Fusion
OLLAMA_MODEL = "mistral"
OLLAMA_TEMP = 0.1
OLLAMA_MAX_TOKENS = 1024
CONFLICT_LOW_THRESHOLD = 0.7
CONFLICT_MEDIUM_THRESHOLD = 0.3

REQUIRED_KEYS = {"stance", "confidence", "conflict_level", "signals", "reasoning"}
REQUIRED_SIGNAL_KEYS = {"news", "chart", "timeseries", "positioning"}

# Feedback
# Weights are per-pair multipliers stored in PostgreSQL weight_log table.
# Each modality confidence is multiplied by its weight before fusion.
# Floor prevents any modality being zeroed out. Ceiling prevents runaway
# boosting on long streaks of correct calls. Learning rate matches the
# prototype (0.1 per feedback event) - conservative to avoid oscillation.
LEARNING_RATE = 0.1
WEIGHT_FLOOR = 0.1
WEIGHT_CEILING = 2.0
DEFAULT_WEIGHT = 1.0
MODALITIES = ["news", "chart", "timeseries", "positioning"]