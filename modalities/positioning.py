import os
import logging
import requests
from dotenv import load_dotenv
from urllib.parse import unquote
from infrastructure.config import MYFXBOOK_URL, MYFXBOOK_LOGIN_URL, MYFXBOOK_LOGOUT_URL, CONTRARIAN_THRESHOLD

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Usage for this file can all be seen in the JSON section of myfxbook official API page https://www.myfxbook.com/api

def login() -> str | None:
    """Authenticates with the MyFxBook API.

    - Use my credentials stored in the project's .env file
    
    Returns session ID if successful else None.
    """
    email = os.getenv("MYFXBOOK_EMAIL")
    password = os.getenv("MYFXBOOK_PASSWORD")

    if not email or not password:
        logger.error("MYFXBOOK_EMAIL or MYFXBOOK_PASSWORD not set in .env file.")
        return None

    try:
        response = requests.post(
            MYFXBOOK_LOGIN_URL,
            data={"email": email, "password": password},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            logger.error(f"MyFxBook login failed: {data.get('message')}")
            return None

        session_id = data.get("session")

        logger.info("MyFxBook login successful.")
        # Decode URL-encoded session token returned by the API
        return unquote(session_id) if session_id else None

    except requests.exceptions.RequestException as e:
        logger.error(f"MyFxBook login request failed: {e}")
        return None

def logout(session_id: str) -> None:
    """Logs out of MyFxBook API to terminate session"""
    try:
        response = requests.get(
            MYFXBOOK_LOGOUT_URL,
            params={"session": session_id},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            logger.warning(f"MyFxBook logout returned error: {data.get('message')}")
        else:
            logger.info("MyFxBook logout successful.")

    except requests.exceptions.RequestException as e:
        logger.error(f"MyFxBook logout request failed: {e}")

def fetch_community_outlook(session_id: str) -> list | None:
    """Fetches retail positioning (community outlook) data from the MyFxBook API.

    Returns list of symbol dicts from the response, else None.
    """
    try:
        response = requests.get(
            MYFXBOOK_URL,
            params={"session": session_id},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            logger.error(f"MyFxBook outlook fetch failed: {data.get('message')}")
            return None

        symbols = data.get("symbols", [])
        logger.info(f"Fetched community outlook for {len(symbols)} symbols.")
        return symbols

    except requests.exceptions.RequestException as e:
        logger.error(f"MyFxBook outlook request failed: {e}")
        return None

def extract_pair_data(symbols: list, pair: str) -> dict | None:
    """Extracts positioning data for the requested currency pair from symbol list.

    Returns the matching symbol dict, else None.
    """
    for symbol in symbols:
        # MyFxBook stores pairs without the '=X' ticker suffix (unlike yfinance)
        if symbol.get("name", "").upper() == pair.upper():
            return symbol

    logger.warning(f"Pair {pair} not found in MyFxBook community outlook response.")
    return None

def apply_contrarian_rule(pair_data: dict) -> dict:
    """Converts retail positioning into a contrarian trading signal.

    - Contrarian rule means that retail traders tend to be wrong at extremes, so we will take the opposite side against them.
      eg. If 70% of community are Long for EURUSD, we will go SHORT

    Returns dict with keys: signal, confidence, long_pct, short_pct, note.
    """
    short_pct = pair_data.get("shortPercentage", 50)
    long_pct = pair_data.get("longPercentage", 50)

    # Majority retail traders are short -> contrarian bullish
    if short_pct >= CONTRARIAN_THRESHOLD:
        signal = 1
        confidence = round(short_pct / 100, 4)
        note = f"Retail {short_pct}% short - contrarian bullish"
    # Majority retail traders are long -> contrarian bearish
    elif long_pct >= CONTRARIAN_THRESHOLD:
        signal = -1
        confidence = round(long_pct / 100, 4)
        note = f"Retail {long_pct}% long - contrarian bearish"
    # No positioning that hits/exceeds threshold detected -> neutral
    else:
        signal = 0
        confidence = 0.5
        note = f"No extreme positioning ({long_pct}% long / {short_pct}% short) - neutral"

    logger.info(f"Positioning signal: {note}")
    return {
        "signal": signal,
        "confidence": confidence,
        "long_pct": long_pct,
        "short_pct": short_pct,
        "note": note
    }

def get_positioning_signal(pair: str) -> dict:
    """Main entry point for the positioning modality.

    - Logs in -> fetches community outlook -> extract pair data -> apply contrarian rule -> logs out

    Returns signal dict with keys: signal, confidence, long_pct, short_pct, note.
    """
    # Default response if the API or requested pair is unavailable
    # * this may happen if you spam the process (log in/out) too many time in short period
    # * to counter this, caching will be used in the orchestrator segment in other file
    neutral_fallback = {
        "signal": 0,
        "confidence": 0.0,
        "long_pct": None,
        "short_pct": None,
        "note": "Fallback neutral - MyFxBook API unavailable or pair not found"
    }

    session_id = login()
    if not session_id:
        logger.warning("Could not obtain MyFxBook session, returning neutral fallback.")
        return neutral_fallback

    try:
        symbols = fetch_community_outlook(session_id)
        if not symbols:
            logger.warning("No symbols returned from MyFxBook, returning neutral fallback.")
            return neutral_fallback

        pair_data = extract_pair_data(symbols, pair)
        if not pair_data:
            logger.warning(f"Pair {pair} not found, returning neutral fallback.")
            return neutral_fallback

        result = apply_contrarian_rule(pair_data)
        logger.info(f"Positioning signal for {pair}: {result}")
        return result

    finally:
        # Always logout regardless of success or failure
        logout(session_id)


# Run smoke test with python -m modalities.positioning
if __name__ == "__main__":
    result = get_positioning_signal("EURUSD")
    print(result)