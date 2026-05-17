import ast
import os

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Movie Explorer",
    page_icon="🎬",
    layout="wide",
)

# ──────────────────────────────────────────────
# Helpers — shared preprocessing utilities
# ──────────────────────────────────────────────

def safe_parse_list(x):
    """Convert a stringified list to a Python list; return [] on failure."""
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return [str(i) for i in parsed] if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def is_valid_movie(row) -> bool:
    """Filter out non-film entries: shorts, trailers, documentaries, stubs."""
    title    = str(row.get("title",    "")).lower()
    overview = str(row.get("overview", "")).lower()
    genres   = str(row.get("genres_list", "")).lower()
    noise    = ["making", "soundtrack", "behind", "short",
                "trailer", "featurette", "episode"]
    if any(w in title or w in overview for w in noise):
        return False
    if "documentary" in genres:
        return False
    if len(overview) < 80:
        return False
    return True


def build_movie_doc(row) -> str:
    """Compose a rich text snippet per movie for embedding."""
    genres   = ", ".join(row["genres_list"]) if row["genres_list"] else "Unknown"
    cast     = " ".join(row["Cast_list"][:3]) if row["Cast_list"] else "Unknown"
    director = str(row.get("Director", "Unknown"))
    overview = str(row.get("overview", ""))[:250]
    return (
        f"Genres: {genres}. "
        f"Director: {director}. "
        f"Cast: {cast}. "
        f"Story: {overview}."
    )


def expand_query(query: str) -> str:
    """Add semantic context to short queries to improve FAISS recall."""
    q = query.lower()
    expansions = {
        "inception": "sci-fi psychological thriller mind-bending Nolan complex narrative",
        "action":    "high intensity action explosions chase combat thriller",
        "romantic":  "love story romance relationship drama heartfelt emotional",
        "horror":    "scary horror suspense fear supernatural thriller",
        "comedy":    "funny comedy humor lighthearted laugh entertaining",
    }
    for kw, expanded in expansions.items():
        if kw in q:
            return expanded
    return query


# ──────────────────────────────────────────────
# Cached resource loaders
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Downloading dataset from Kaggle…")
def load_dataset():
    """Download and quality-filter the 1M movie dataset."""
    import kagglehub

    path_big = kagglehub.dataset_download(
        "shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m"
    )
    raw_csv = f"{path_big}/IMDB TMDB Movie Metadata Big Dataset (1M).csv"

    RAG_COLS = [
        "title", "vote_average", "vote_count", "popularity",
        "runtime", "revenue", "budget", "release_year",
        "genres_list", "Cast_list", "Director", "overview",
    ]
    df = pd.read_csv(raw_csv, usecols=RAG_COLS)
    df = df[df["vote_count"] > 1000].copy()

    df = df[df.apply(is_valid_movie, axis=1)].copy()
    df.dropna(subset=["title", "overview"], inplace=True)

    df["genres_list"] = df["genres_list"].apply(safe_parse_list)
    df["Cast_list"]   = df["Cast_list"].apply(safe_parse_list)
    df["Director"]    = df["Director"].fillna("Unknown")
    df["overview"]    = df["overview"].fillna("")

    # Keep only rows with at least one genre
    df = df[df["genres_list"].map(len) > 0].reset_index(drop=True)

    df["movie_doc"] = df.apply(build_movie_doc, axis=1)
    return df


@st.cache_resource(show_spinner="Building FAISS index (first run only)…")
def build_index(_df):
    """Embed movie documents and build a FAISS cosine similarity index."""
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    docs = _df["movie_doc"].tolist()
    embeddings = embedder.encode(
        docs,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        device=device,
    )
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index


@st.cache_resource(show_spinner="Loading Gemini…")
def load_gemini():
    """Configure and return the best available Gemini Flash model name."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, None

    genai.configure(api_key=api_key)
    valid_model_name = None
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            if "flash" in m.name:
                valid_model_name = m.name
                break
            elif valid_model_name is None:
                valid_model_name = m.name

    return genai, valid_model_name


# ──────────────────────────────────────────────
# Core recommendation logic
# ──────────────────────────────────────────────

def retrieve_movies(query: str, df, embedder, index, k: int = 10) -> list:
    """Retrieve top-k movies via FAISS + hybrid re-ranking."""
    import faiss

    query_vec = embedder.encode([expand_query(query)], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k * 5)

    pop_max = df["popularity"].max()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        final_score = (
            0.70 * float(score)
            + 0.20 * (row["vote_average"] / 10)
            + 0.10 * (row["popularity"] / pop_max)
        )
        results.append({
            "title":       row["title"],
            "final_score": final_score,
            "rating":      row["vote_average"],
            "genres":      ", ".join(row["genres_list"]),
            "director":    row["Director"],
            "cast":        ", ".join(row["Cast_list"][:3]),
            "overview":    row["overview"][:300],
        })
    return sorted(results, key=lambda x: x["final_score"], reverse=True)


def diversify(results: list) -> list:
    """Remove near-duplicate titles (e.g. sequels sharing the same root name)."""
    seen, final = set(), []
    for r in results:
        key = r["title"].split(":")[0].strip().lower()
        if key not in seen:
            seen.add(key)
            final.append(r)
    return final


def recommend(query: str, df, embedder, index, k: int = 5) -> list:
    """Public recommend API: retrieve, diversify, top-k."""
    return diversify(retrieve_movies(query, df, embedder, index, k=k * 3))[:k]


SYSTEM_PROMPT = (
    "You are an expert movie recommendation assistant. "
    "Recommend ONLY from the movies provided in the context. "
    "Never invent movies not in the list. "
    "For each recommendation, explain why it matches the user request "
    "based on the overview, genre, and rating."
)


def format_context(movies: list) -> str:
    lines = []
    for i, m in enumerate(movies, 1):
        lines.append(
            f"{i}. {m['title']}  |  Rating: {m['rating']}/10  |  Genres: {m['genres']}\n"
            f"   Overview: {m['overview']}"
        )
    return "\n\n".join(lines)


def gemini_explain(query: str, movies: list, genai, model_name: str) -> str:
    """Ask Gemini to explain the retrieved movie recommendations."""
    context  = format_context(movies)
    prompt   = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser request: {query}"
    model    = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

def main():
    # ── Header ──────────────────────────────────────────
    st.title("🎬 AI Movie Explorer")
    st.markdown(
        "Search through **1M+ movies** using semantic AI search, "
        "powered by FAISS and Google Gemini."
    )

    # ── Sidebar — API key ────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key_input = st.text_input(
            "Gemini API Key",
            value=os.environ.get("GEMINI_API_KEY", "AIzaSyC3tXuXuzkCXsWbYyOHo8YIPli9iu-we28"),
            type="password",
            help="Get yours at https://aistudio.google.com",
        )
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input

        st.divider()
        st.markdown("**About this app**")
        st.markdown(
            "- 🔍 FAISS semantic search\n"
            "- 🤖 Google Gemini explanations\n"
            "- 📊 Hybrid re-ranking (similarity + rating + popularity)\n"
            "- 🎯 Diversity filter (no duplicate franchises)"
        )
        k = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    # ── Load resources ────────────────────────────────────
    with st.spinner("Loading movie database…"):
        df = load_dataset()

    with st.spinner("Building search index…"):
        embedder, index = build_index(df)

    gemini_available = bool(os.environ.get("GEMINI_API_KEY", "AIzaSyC3tXuXuzkCXsWbYyOHo8YIPli9iu-we28"))
    if gemini_available:
        genai_module, model_name = load_gemini()
    else:
        genai_module, model_name = None, None

    if not gemini_available:
        st.warning(
            "Gemini API key not set. Results will show matched movies without "
            "AI-generated explanations. Enter your key in the sidebar to enable full RAG."
        )

    st.divider()

    # ── Search bar ────────────────────────────────────────
    query = st.text_input(
        "🔎 Describe what you want to watch:",
        placeholder="e.g. Mind-bending sci-fi thrillers with a twist ending",
    )

    example_queries = [
        "Romantic comedies set in New York",
        "Action movies with female leads",
        "Psychological thrillers like Inception",
        "Award-winning dramas from the 90s",
        "Animated family adventure movies",
    ]

    st.markdown("**Try an example:**")
    cols = st.columns(len(example_queries))
    for col, example in zip(cols, example_queries):
        if col.button(example, use_container_width=True):
            query = example

    if not query:
        st.stop()

    # ── Run recommendation ────────────────────────────────
    with st.spinner("Searching movies…"):
        movies = recommend(query, df, embedder, index, k=k)

    if not movies:
        st.error("No matching movies found. Try a different description.")
        st.stop()

    # ── Gemini explanation ────────────────────────────────
    if gemini_available and genai_module and model_name:
        with st.spinner("Generating AI explanation…"):
            explanation = gemini_explain(query, movies, genai_module, model_name)

        st.subheader("🤖 AI Recommendation")
        st.markdown(explanation)
        st.divider()

    # ── Movie cards ───────────────────────────────────────
    st.subheader(f"🎥 Top {len(movies)} matches for: *{query}*")

    for i, movie in enumerate(movies, 1):
        with st.expander(
            f"**{i}. {movie['title']}** — ⭐ {movie['rating']:.1f}/10",
            expanded=(i <= 3),
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Overview:** {movie['overview']}")
            with col2:
                st.markdown(f"**Genres:** {movie['genres']}")
                st.markdown(f"**Director:** {movie['director']}")
                if movie["cast"]:
                    st.markdown(f"**Cast:** {movie['cast']}")
                st.markdown(f"**Match score:** `{movie['final_score']:.3f}`")


if __name__ == "__main__":
    main()
