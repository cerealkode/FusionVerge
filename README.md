# ◈ FusionVerge

**A multi-modal trading decision support system that reconciles price action, chart patterns, news sentiment and retail positioning into one explainable trading decision. And unlike comparable research systems, it surfaces *how much the signals disagreed* rather than resolving conflict silently.**

*CM3070 Final Year Project - University of London, Dept. of Computer Science*

📺 [Watch the demo on YouTube](https://youtu.be/sjHvguP1Dlg) &nbsp;·&nbsp; 📄 [Read the full report](./Final_report.pdf)

---

## The Problem

Retail traders don't lack data: price charts, news, sentiment feeds, and positioning stats are all freely available. What's missing is a way to combine them. Each source is checked separately, weighed against the others with no formal process, and the resulting decision often has more to do with which tool the trader happened to check last than with actual market conditions.

This isn't just an inconvenience, the research backs it up. Adding more data sources *without* a way to structure them has been shown to actively worsen decision quality, not improve it. And individual tools are shakier than they look in isolation: technical indicators are well-documented as subjective enough that the same chart can lead two analysts to opposite conclusions.

<p align="center">
  <img src="./assets/problem.png" alt="Problem motivation" width="80%" />
</p>

FusionVerge's answer is to stop treating this as a single-model problem. Four specialized models process each data type independently, and an LLM-based fusion layer reconciles them, explicitly reporting *how much they agreed*, not just a final answer. This is also where it departs from comparable multi-modal trading systems in recent research (eg. FinAgent, MountainLion): those resolve modality conflict internally and never expose it, ending in a bare trading action or a narrative report. FusionVerge treats conflict as a first-class output the user can actually see.

---

## How It Works

<p align="center">
  <img src="./assets/system-overview.png" alt="System overview" width="90%" />
</p>

1. **Multi-modal inputs:** OHLC price series, a rendered candlestick chart image, scraped news headlines, and retail positioning data are pulled for the selected currency pair.
2. **Specialized models per modality:** each input type is handled by a model suited to it (see below).
3. **Fusion layer:** a local LLM (Mistral, via Ollama) receives all four modality outputs plus a pre-computed conflict score, and reconciles them into one stance with reasoning.
4. **Structured output:** stance (bullish/bearish/neutral), confidence, conflict level, per-modality explanation, and the LLM's reasoning.
5. **Feedback loop:** the user marks the outcome (correct/incorrect/uncertain) after the fact. Modality weights are adjusted per currency pair based on whether that modality agreed with the outcome, so the system's trust in each modality evolves over time.

---

## The Four Modalities

| Modality | Model / Method | What it does |
|---|---|---|
| **News Sentiment** | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) | Scrapes forex headlines (FXStreet, ForexLive RSS), classifies sentiment, then maps it to a directional signal per currency pair (eg. positive USD news → bearish EURUSD) |
| **Chart Pattern** | [YOLOv8 pattern detector](https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8) | Renders a candlestick chart from live OHLC data and runs object detection to identify classical chart patterns (head & shoulders, double tops/bottoms, triangles) |
| **Time-Series** | Custom-trained GRU (Keras/TensorFlow) | Predicts short-term price direction from OHLC + technical indicators (RSI, MACD, EMA, ATR), using the best-performing feature set/lookback found per pair-timeframe during training |
| **Retail Positioning** | Rule-based contrarian logic | Pulls community long/short percentages from the MyFxBook API and applies a contrarian rule: retail crowds tend to be wrong at extremes |

Each modality outputs a `signal` (+1 / -1 / 0) and a `confidence` score independently; no modality knows what the others produced.

---

## Fusion Layer

Before fusion, each modality's confidence is scaled by a **learned per-pair weight** (see Feedback Loop below). A conflict score is computed separately from confidence-weighted signal imbalance across modalities, then passed to the LLM alongside all four modality outputs.

The LLM (Mistral via Ollama, run locally) is prompted to return strict structured JSON:

```json
{
  "stance": "bullish",
  "confidence": 0.74,
  "conflict_level": "low",
  "signals": {
    "news": "...",
    "chart": "...",
    "timeseries": "...",
    "positioning": "..."
  },
  "reasoning": "..."
}
```

Output is validated against a required schema and retried up to 3 times on malformed/invalid JSON (using `json_repair` to recover from near-miss formatting), falling back to a neutral default if all retries fail.

<details>
<summary>Fusion layer sequence diagram (conflict scoring -> prompt -> LLM call -> validation -> retry)</summary>

<p align="center">
  <img src="./assets/fusion-sequence-diagram.png" alt="Fusion sequence diagram" width="70%" />
</p>

</details>

---

## Feedback Loop

This is what separates FusionVerge from a static ensemble: **trust in each modality is learned per currency pair, not fixed.**

- After an analysis, the user can mark the outcome as correct, incorrect, or uncertain.
- Modalities that agreed with a *correct* outcome are rewarded (weight increases slightly).
- Modalities that agreed with an *incorrect* outcome are penalized.
- Modalities that *disagreed* with a wrong consensus are rewarded (they were the dissenting voice that should've been trusted more).
- Weights are bounded (floor/ceiling) to prevent runaway trust or a modality being zeroed out entirely, and every update is versioned in `weight_log` so the trajectory is visible on the dashboard.

Nothing updates on unanimous-but-wrong calls: there's no differential signal to attribute blame to.

---

## Evaluation

The project tested three hypotheses against the full pipeline (see the final report for full methodology):

| Hypothesis | Result |
|---|---|
| **H1**: Multi-modal fusion outperforms a single unorchestrated LLM | **Supported.** Feeding raw data straight to Mistral with no specialist pipeline failed outright (JSON parsing errors) on 4/5 test pairs across repeated runs, and produced generic, templated reasoning even when it succeeded. The full pipeline never failed and averaged a coherence score of 4.8/5 vs. 3.3/5 for the naive baseline. |
| **H2**: Explicit conflict modeling improves interpretability | **Supported.** Conflict-level output tracked directly with how the LLM explained its stance: high-conflict outputs explicitly named which modality's influence was discounted and why, rather than presenting a flat, hedged summary. |
| **H3**: The feedback loop improves output consistency over time | **Partially supported.** A 10-run comparison of default vs. post-feedback weights showed stance consistency rising from 60% to 80% and confidence variance narrowing, but a second comparison under low-conflict conditions showed no measurable change (ceiling effect); encouraging, not conclusive. |

**Time-series model (GRU) accuracy sweep:** 224 models were trained across all 7 pairs x 2 timeframes x 16 hyperparameter configurations to select the best config per pair/timeframe:

- **Daily timeframe:** 80-85% directional accuracy on 6/7 pairs
- **Hourly timeframe:** 53-58% directional accuracy across the board, barely above random

This is a genuine (if slightly inconvenient) finding: daily price action carries a meaningfully stronger learnable signal than hourly data for this model class, which the report flags as an open item the dashboard doesn't yet surface to the user.

Full unit/integration test suite: **266 tests across 8 modules**, including deliberate failure-injection tests (bad credentials, malformed LLM output, missing model files) to confirm the pipeline degrades to neutral rather than crashing.

---

## Tech Stack

- **Frontend:** Streamlit
- **Sentiment:** HuggingFace Transformers (FinBERT)
- **Chart detection:** Ultralytics YOLOv8
- **Time-series:** TensorFlow/Keras (GRU), pandas-ta for indicators
- **Fusion LLM:** Ollama (Mistral, locally hosted)
- **Data:** yfinance (OHLC), feedparser (RSS), MyFxBook API (positioning)
- **Persistence:** PostgreSQL (feedback + weight history)

---

## Project Structure

The codebase is layered by dependency direction: `infrastructure` sits at the base with zero internal dependencies, `modalities` builds on it, `core` orchestrates the modalities and runs fusion, and `ui` renders it all in `main.py`. Each modality module (`sentiment.py`, `chart.py`, `timeseries.py`, `positioning.py`) exposes one entry point, so swapping a model only touches that file.

<details>
<summary>Module dependency diagram</summary>

<p align="center">
  <img src="./assets/dependency-structure.png" alt="Dependency structure" width="90%" />
</p>

</details>

---

## Running It Locally

**Prerequisites:**
- Python 3.10+
- PostgreSQL instance (tables are created automatically on first run if they don't exist)
- [Ollama](https://ollama.com) installed locally with the `mistral` model pulled (`ollama pull mistral`)
- Trained GRU models + scalers + `best_configs.json` in `models/` (produced by the training notebook; not all pairs/timeframes may be included in this repo)
- A MyFxBook account (free) for the positioning modality

**Setup:**

```bash
git clone https://github.com/<cerealcode>/fusionverge.git
cd fusionverge
pip install -r requirements.txt
```

Create a `.env` file:

```
DATABASE_URL=postgresql://user:password@localhost:5432/fusionverge
MYFXBOOK_EMAIL=your_email
MYFXBOOK_PASSWORD=your_password
```

Run it:

```bash
streamlit run main.py
```

The database schema is created automatically on first launch. FinBERT and the YOLO model are downloaded on first run and cached for subsequent sessions.

> **Note:** If a modality's model/data isn't available (eg. missing GRU model for a pair-timeframe, MyFxBook rate-limited), the pipeline falls back to a neutral signal for that modality rather than crashing, so the fusion layer always gets four inputs, even in degraded conditions.

---

## Demo

📺 [Quick video walkthrough on YouTube](https://youtu.be/sjHvguP1Dlg)

---

## Notes

This was built as a final year project. The full report (problem motivation, related work, evaluation methodology, and results) is included in this repo for anyone who wants the in-depth version of what's summarized here.