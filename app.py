"""
K-Drama Recommendation Tool - Web Backend
Serves the recommender via REST API for the web UI.
"""

import os
import math
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from kdrama_recommender import KDramaRecommender

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Load recommender once at startup
RECOMMENDER = None
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATA_DIR, "kdrama_DATASET.csv")


def get_recommender():
    """Lazy-load and cache the recommender (loads data + prepares features)."""
    global RECOMMENDER
    if RECOMMENDER is None:
        RECOMMENDER = KDramaRecommender()
        if not os.path.isfile(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
        RECOMMENDER.load_and_prepare_data(DATASET_PATH)
        # Match column names to kdrama_DATASET.csv (Title, Year of release, Description, Actors)
        RECOMMENDER.prepare_features(
            name_col="Title",
            genre_col="Genre",
            year_col="Year of release",
            rating_col="Rating",
            episodes_col="Number of Episodes",
            cast_col="Actors",
            synopsis_col="Description",
            tags_col="Tags",
        )
    return RECOMMENDER


def _safe_float(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _safe_int(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def df_row_to_drama(row, rec, extra_cols=None):
    """Convert a dataframe row (Series) to a JSON-friendly drama object."""
    name_col = rec._find_name_column()
    rating_col = rec._find_rating_column()
    genre_col = rec._find_genre_column()
    year_col = rec._find_year_column()
    tags_col = rec._find_tags_column()
    out = {
        "name": str(row.get(name_col, "")),
        "rating": _safe_float(row.get(rating_col)) if rating_col else None,
        "genre": str(row.get(genre_col, "")) if genre_col else None,
        "year": _safe_int(row.get(year_col)) if year_col else None,
        "tags": str(row.get(tags_col, "")) if tags_col else None,
    }
    if extra_cols:
        for k, v in extra_cols.items():
            out[k] = v
    return out


@app.route("/")
def index():
    """Serve the main web UI."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    """Health check."""
    return jsonify({"status": "ok"})


@app.route("/api/dramas")
def list_dramas():
    """List drama names for search/autocomplete."""
    try:
        rec = get_recommender()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    name_col = rec._find_name_column()
    names = rec.dramas_df[name_col].dropna().astype(str).unique().tolist()
    return jsonify({"dramas": sorted(names)})


@app.route("/api/similar", methods=["POST"])
def similar():
    """Get dramas similar to a given drama."""
    try:
        rec = get_recommender()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    data = request.get_json(force=True, silent=True) or {}
    drama_name = (data.get("drama_name") or "").strip()
    n = max(1, min(20, int(data.get("n", 10))))
    if not drama_name:
        return jsonify({"error": "drama_name is required"}), 400
    similar_df = rec.get_similar_dramas(drama_name, n_recommendations=n)
    if similar_df.empty:
        return jsonify({"error": f"No drama found matching '{drama_name}'", "recommendations": []}), 200
    results = []
    for _, row in similar_df.iterrows():
        results.append(df_row_to_drama(row, rec, {"similarity_score": round(float(row["similarity_score"]), 4)}))
    return jsonify({"recommendations": results})


@app.route("/api/by-preferences", methods=["POST"])
def by_preferences():
    """Get recommendations by genre/tag/rating/year preferences."""
    try:
        rec = get_recommender()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    data = request.get_json(force=True, silent=True) or {}
    preferred_genres = data.get("genres") or []
    if isinstance(preferred_genres, str):
        preferred_genres = [g.strip() for g in preferred_genres.split(",") if g.strip()]
    preferred_tags = data.get("tags") or []
    if isinstance(preferred_tags, str):
        preferred_tags = [t.strip() for t in preferred_tags.split(",") if t.strip()]
    min_rating = data.get("min_rating")
    if min_rating is not None:
        try:
            min_rating = float(min_rating)
        except (TypeError, ValueError):
            min_rating = None
    year_min = data.get("year_min")
    year_max = data.get("year_max")
    year_range = None
    if year_min is not None and year_max is not None:
        try:
            year_range = (int(year_min), int(year_max))
        except (TypeError, ValueError):
            pass
    n = max(1, min(20, int(data.get("n", 10))))
    prefs_df = rec.get_recommendations_by_preferences(
        preferred_genres=preferred_genres or None,
        preferred_tags=preferred_tags or None,
        min_rating=min_rating,
        year_range=year_range,
        n_recommendations=n,
    )
    results = []
    for _, row in prefs_df.iterrows():
        extra = {}
        if "preference_score" in row.index:
            extra["preference_score"] = round(float(row["preference_score"]), 2)
        results.append(df_row_to_drama(row, rec, extra))
    return jsonify({"recommendations": results})


@app.route("/api/genres")
def genres():
    """List unique genres for filter UI."""
    try:
        rec = get_recommender()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    genre_col = rec._find_genre_column()
    if not genre_col:
        return jsonify({"genres": []})
    parts = rec.dramas_df[genre_col].dropna().astype(str).str.split(",").explode()
    genres_list = sorted({p.strip() for p in parts if p.strip()})
    return jsonify({"genres": genres_list})


@app.route("/api/tags")
def tags():
    """List unique tags for filter UI."""
    try:
        rec = get_recommender()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    tags_col = rec._find_tags_column()
    if not tags_col:
        return jsonify({"tags": []})
    parts = rec.dramas_df[tags_col].dropna().astype(str).str.split(",").explode()
    tags_list = sorted({p.strip() for p in parts if p.strip()})
    return jsonify({"tags": tags_list})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
