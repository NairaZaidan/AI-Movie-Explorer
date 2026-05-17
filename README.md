# 🎬 AI Movie Explorer

A RAG-based (Retrieval-Augmented Generation) movie recommendation system that combines semantic search with Google Gemini to give natural-language movie recommendations from a dataset of 1M+ films.

---

## Features

- **Semantic search** over 1M+ movies using FAISS and SentenceTransformer embeddings
- **Hybrid re-ranking** — balances semantic similarity, vote average, and popularity
- **Diversity filter** — avoids recommending duplicate franchises/sequels
- **Google Gemini** generates personalised explanations for each recommendation
- **ML models** — XGBoost rating predictor and Random Forest genre classifier
- **Sentiment analysis** — DistilBERT scores 50K IMDB reviews from −1 to +1
- **Gradio UI** (notebook) and **Streamlit app** (`app.py`)

---

## Project Structure

```
.
├── AI_Movie_Explorer.ipynb   # Full research notebook (exploration → training → RAG)
├── app.py                    # Streamlit production app
├── README.md                 # This file
```

---

## Datasets

Both datasets are downloaded automatically via `kagglehub`:

| Dataset | Purpose |
|---------|---------|
| [`shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m`](https://www.kaggle.com/datasets/shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m) | Movie metadata — ratings, genres, cast, director, overview |
| [`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 50K text reviews with positive/negative sentiment labels |

---

## Notebook Pipeline

| Section | Description |
|---------|-------------|
| 0 | Imports & setup |
| 1 | Dataset download via `kagglehub` |
| 2 | Data cleaning & feature engineering |
| 3 | ML model training (XGBoost + Random Forest) |
| 4 | Sentiment analysis with DistilBERT |
| 5 | RAG pipeline — embeddings, FAISS index, retriever |
| 6 | Gemini LLM integration & end-to-end test |
| 7 | Gradio UI |

---

## Quick Start

### 1. Install dependencies

```bash
pip install kagglehub pandas numpy scikit-learn xgboost transformers \
            sentence-transformers faiss-cpu torch google-generativeai \
            gradio streamlit
```

### 2. Set up Kaggle credentials

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Or place `~/.kaggle/kaggle.json`:
```json
{"username": "your_username", "key": "your_api_key"}
```

### 3. Set your Gemini API key

```bash
export GEMINI_API_KEY=your_gemini_api_key
```

Get a free key at [https://aistudio.google.com](https://aistudio.google.com).

### 4a. Run the Streamlit app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### 4b. Run the notebook

Open `AI_Movie_Explorer.ipynb` in Jupyter or Google Colab and run all cells in order.

For Colab, set your Gemini key via **Secrets** (the 🔑 icon in the left sidebar) instead of hardcoding it.

---

## Architecture

```
User query
    │
    ▼
expand_query()          ← adds semantic context to short queries
    │
    ▼
SentenceTransformer     ← encodes query to 384-dim vector
    │
    ▼
FAISS IndexFlatIP       ← cosine similarity search over ~50K movie vectors
    │
    ▼
Hybrid re-ranking       ← 70% semantic + 20% rating + 10% popularity
    │
    ▼
Diversity filter        ← removes franchise duplicates
    │
    ▼
Google Gemini Flash     ← generates natural-language explanation
    │
    ▼
Streamlit / Gradio UI
```

---

## ML Models

### XGBoost Regressor
- **Target:** `vote_average` (predicted rating 0–10)
- **Features:** scaled numeric fields + genre one-hot encodings + encoded director/cast
- **Hyperparameters:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=6`

### Random Forest Classifier
- **Target:** `main_genre` (primary genre label)
- **Features:** same feature matrix as regressor
- **Hyperparameters:** `n_estimators=50`

---

## Preprocessing Summary

1. **Column selection** — 12 relevant columns from the raw CSV
2. **Null handling** (all in one pass, before any encoding):
   - Drop rows with no `overview`
   - Fill `release_year` / `runtime` with median
   - Fill `budget` / `revenue` with 0
   - Fill `Director` with `"Unknown"`
3. **List parsing** — `safe_parse_list()` safely converts stringified lists
4. **Genre filter** — drop rows with empty genre lists
5. **Multi-label encoding** — `MultiLabelBinarizer` for `genres_list`
6. **Label encoding** — `LabelEncoder` for `Director` and `main_genre`
7. **Scaling** — `MinMaxScaler` on numeric features

---