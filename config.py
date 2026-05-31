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

# Supported timeframes (yfinance interval strings)
SUPPORTED_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]

# History period per timeframe (yfinance period strings)
# Shorter timeframes have less available history on yfinance
TIMEFRAME_HISTORY = {
    "1h": "2y",
    "4h": "2y",
    "1d": "5y",
}

# Sentiment
SENTIMENT_MODEL = "ProsusAI/finbert"
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# Two RSS feeds used for coverage and redundancy
RSS_FEEDS = [
    "https://www.fxstreet.com/rss/news",
    "https://www.forexlive.com/feed/news",
]

# Keywords for filtering headlines by asset
# Known limitation: USD appears across all pairs, Fed headlines will match multiple pairs
# Slang terms (greenback, cable, etc.) included to reduce coverage gaps
ASSET_KEYWORDS = {
    "EURUSD": ["EUR", "USD", "euro", "dollar", "EURUSD", "Fed", "ECB", "eurozone", "greenback"],
    "GBPUSD": ["GBP", "USD", "pound", "sterling", "cable", "GBPUSD", "BOE", "Fed", "greenback"],
    "AUDUSD": ["AUD", "USD", "aussie", "dollar", "AUDUSD", "RBA", "Fed", "greenback"],
    "NZDUSD": ["NZD", "USD", "kiwi", "dollar", "NZDUSD", "RBNZ", "Fed", "greenback"],
    "USDJPY": ["USD", "JPY", "yen", "dollar", "USDJPY", "BOJ", "Fed", "greenback"],
    "USDCHF": ["USD", "CHF", "franc", "dollar", "USDCHF", "SNB", "Fed", "greenback"],
    "USDCAD": ["USD", "CAD", "loonie", "dollar", "USDCAD", "BOC", "Fed", "greenback", "oil"],
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

# Chart
CHART_STYLE = "charles"
CHART_DPI = 150
CHART_SAVEPATH = "chart_output.png"
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
LOOKBACK = 15
FEATURES_BASE = ["Open", "High", "Low", "Close", "Volume"]
FEATURES_EXTRA = ["RSI", "MACD", "EMA"]
GRU_UNITS_1 = 50
GRU_UNITS_2 = 25
GRU_DENSE_UNITS = 20
GRU_DROPOUT = 0.2
GRU_LR = 0.001
GRU_BATCH_SIZE = 32
GRU_EPOCHS = 50
GRU_PATIENCE = 15
GRU_SAVEPATH = "models/gru_eurusd_1h.keras" # may require things like models/gru_{pair}_{timeframe}.keras during runtime for later. change accrodingly proabbly

# Positioning
MYFXBOOK_URL = "https://www.myfxbook.com/api/get-community-outlook.json"
CONTRARIAN_THRESHOLD = 60

# Fusion
OLLAMA_MODEL = "mistral"
OLLAMA_TEMP = 0.1
OLLAMA_MAX_TOKENS = 1024
CONFLICT_LOW_THRESHOLD = 0.7
CONFLICT_MEDIUM_THRESHOLD = 0.3

# Feedback
LEARNING_RATE = 0.1
WEIGHT_FLOOR = 0.1
DEFAULT_WEIGHT = 1.0
SQLITE_DB_PATH = "feedback.db"