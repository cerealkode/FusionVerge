import feedparser
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from infrastructure.config import SENTIMENT_MODEL, RSS_FEEDS, CURRENCY_KEYWORDS, PAIR_CURRENCY_EFFECT, SUPPORTED_PAIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_sentiment_model():
    """Loads ProsusAI/finbert sentiment for model pipeline.

    - Tries auto pipeline method first. falls back to manual load method if that fails.
    - Both method are shown in model usage section in HuggingFace page: https://huggingface.co/ProsusAI/finbert?library=transformers
    """
    try:
        logger.info("Loading FinBERT via auto pipeline...")
        # top_k=None make sure all class scores are returned (positive/negative/neutral), and not just top ones
        pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, top_k=None)
        logger.info("Auto pipeline load successful.")
        return pipe
    except Exception as e:
        logger.warning(f"Auto load failed ({e}), falling back to manual load...")
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None)
        logger.info("Manual load successful.")
        return pipe

def fetch_headlines(pair: str, max_headlines: int = None) -> list[str]:
    """Fetches headlines from all configured RSS feeds and filter by asset keywords.

    - Builds union of both currencies keyword lists so a headline is captured if it mentions either leg of the pair
      eg. pure USD headline is still relevant to EURUSD even if EUR is not mentioned
    - Rejects unsupported pairs early via SUPPORTED_PAIRS before any feed fetching.

    Returns list of matching headline strings.
    """
    if pair not in SUPPORTED_PAIRS:
        logger.warning(f"Pair {pair} not supported, returning empty.")
        return []

    base_currency = pair[:3]
    quote_currency = pair[3:]

    # Combine currency keyword of either leg so either can be matched (eg. EUR + USD)
    keywords = list(set(CURRENCY_KEYWORDS[base_currency] + CURRENCY_KEYWORDS[quote_currency]))

    headlines = []
    seen_titles = set() # normalized titles already added, for cross-feed deduplication
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get("title", "")
                # Case-insensitive keyword matching for headline
                if any(kw.lower() in title.lower() for kw in keywords):
                    normalized = title.strip().lower()
                    if normalized in seen_titles:
                        continue # skip duplicate, eg. same story on 2 feeds
                    seen_titles.add(normalized)
                    headlines.append(title)
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")

    # Cap number of returned headlines (set in function input)
    if max_headlines is not None:
        headlines = headlines[:max_headlines]

    logger.info(f"Fetched {len(headlines)} filtered headlines for {pair}.")
    return headlines

def classify_headlines(pipe, headlines: list[str]) -> list[dict]:
    """Runs FinBERT inference on each headline for sentiment (positive/negative/neutral).

    Return list of dicts with keys: headline, label, confidence.
    """
    results = []
    for text in headlines:
        try:
            output = pipe(text)[0] # remove the outer batch wrapper from List[List[dict]] -> List[dict]
            # Sort descending to get highest confidence prediction
            # * many ways to do this, i use lambda, sort methods can be referenced here https://mimo.org/glossary/python/list-sort()
            top = sorted(output, key=lambda x: x["score"], reverse=True)[0]
            results.append({
                "headline": text,
                "label": top["label"].lower(), # simple normalization
                "confidence": round(top["score"], 4) # shave the decimal numbers for readability
            })
        except Exception as e:
            logger.error(f"FinBERT inference failed for headline '{text}': {e}")
    return results

def detect_currency_focus(headline: str, pair: str) -> str:
    """Detects which currency in the pair the headline focus on/mainly affects.

    - Use simple keyword frequency matching method (more occurrence = more focus)

    Returns base or quote currency code, or "neutral" if not clear.
    """
    base_currency = pair[:3]
    quote_currency = pair[3:]

    base_keywords = CURRENCY_KEYWORDS.get(base_currency, [])
    quote_keywords = CURRENCY_KEYWORDS.get(quote_currency, [])
    headline_lower = headline.lower()

    # Count keyword matches for each currency (simple frequency heuristic)
    base_hits = sum(1 for kw in base_keywords if kw.lower() in headline_lower)
    quote_hits = sum(1 for kw in quote_keywords if kw.lower() in headline_lower)

    # More keyword hit count indicates the focus
    if base_hits > quote_hits:
        return base_currency
    elif quote_hits > base_hits:
        return quote_currency
    else:
        return "neutral" # if tied

def apply_currency_adjustment(classified: list[dict], pair: str) -> list[dict]:
    """Maps raw FinBERT sentiment to a directional signal for the specific pair.

    - FinBERT predicts headline sentiment not pair direction
    - Converts that sentiment into a bullish (+1), bearish (-1), or neutral (0)
      "signal" key for the specified currency pair
    - Eg. positive news for USD means bearish for EURUSD
    
    Returns dict with added signal key.
    """
    pair_effect = PAIR_CURRENCY_EFFECT.get(pair, {})
    adjusted = []

    for item in classified:
        label = item["label"]
        currency_focus = detect_currency_focus(item["headline"], pair)

        if label == "neutral" or currency_focus == "neutral":
            # Skip neutral sentiments as give no signal
            signal = 0
        else:
            # Map currency to directional effect on pair (+1, -1)
            direction = pair_effect.get(currency_focus, 0)
            # Positive sentiment follows direction and negative will flip it
            if label == "positive":
                signal = direction
            elif label == "negative":
                signal = -direction
            else:
                signal = 0

        adjusted.append({**item, "signal": signal})

    return adjusted

def aggregate_signal(adjusted: list[dict]) -> dict:
    """Aggregates per-headline signals into a single modality output (directional).

    - Use confidence-weighted signal mean instead of naive aggregation (higher-confidence ones contribute more)
    - Apply ±0.1 threshold to avoid weak/indecisive signals

    Returns dict with keys: signal (+1/-1/0), confidence and distribution.
    """
    if not adjusted:
        logger.warning("No adjusted headlines to aggregate, returning neutral.")
        return {"signal": 0, "confidence": 0.0, "distribution": {"positive": 0, "negative": 0, "neutral": 0}}

    distribution = {"positive": 0, "negative": 0, "neutral": 0}
    for item in adjusted:
        label = item["label"]
        if label in distribution:
            distribution[label] += 1

    # Weighted average of signals using model confidence
    weighted_sum = sum(item["signal"] * item["confidence"] for item in adjusted)
    total_conf = sum(item["confidence"] for item in adjusted)
    weighted_avg = weighted_sum / total_conf if total_conf > 0 else 0.0

    # Thresholding to avoid weak signals near zero
    if weighted_avg > 0.1:
        signal = 1
    elif weighted_avg < -0.1:
        signal = -1
    else:
        signal = 0

    # Directional conviction - agreement strength across headlines (not model confidence)
    # * abs(weighted_avg) clamped to 1.0 as a safety bound
    directional_confidence = round(min(abs(weighted_avg), 1.0), 4)

    return {
        "signal": signal,
        "confidence": directional_confidence,
        "distribution": distribution
    }

def get_sentiment_signal(pair: str, pipe=None, max_headlines: int = None) -> dict:
    """Main entry point for the sentiment modality.

    - Runs full pipeline: load -> fetch -> classify -> currency adjust -> aggregate
    - Accepts optional preloaded pipeline to avoid repeated model loads

    Returns aggregated signal dict with keys: signal, confidence and distribution.
    """
    if pipe is None:
        pipe = load_sentiment_model()

    headlines = fetch_headlines(pair, max_headlines=max_headlines)

    if not headlines:
        logger.warning(f"No headlines found for {pair}, returning neutral signal.")
        return {"signal": 0, "confidence": 0.0, "distribution": {"positive": 0, "negative": 0, "neutral": 0}}

    classified = classify_headlines(pipe, headlines)
    adjusted = apply_currency_adjustment(classified, pair)
    result = aggregate_signal(adjusted)

    logger.info(f"Sentiment signal for {pair}: {result}")
    return result


# Run smoke test with python -m modalities.sentiment
if __name__ == "__main__":
    result = get_sentiment_signal("EURUSD")
    print(result)