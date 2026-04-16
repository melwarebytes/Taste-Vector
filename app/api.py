"""TasteVector API entry point."""

import os
import sys

# Make src/ importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
from flask import Flask, jsonify, request

from data_loader import load_all
from matrix_builder import build_rating_matrix
from recommender import get_recommendations

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load data once at startup ─────────────────────────────────────────────────

_data_dir = os.path.join(_root, "data")
restaurants, users, ratings = load_all(_data_dir)
R = build_rating_matrix(users, restaurants, ratings)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/restaurants")
def get_restaurants():
    return jsonify(restaurants.to_dict(orient="records"))


@app.post("/rate")
def rate():
    body = request.get_json(force=True)
    user_id       = body.get("user_id")
    restaurant_id = body.get("restaurant_id")
    rating_value  = body.get("rating")

    if user_id is None or restaurant_id is None or rating_value is None:
        return jsonify({"error": "user_id, restaurant_id, and rating are required"}), 400

    uid_list = users["user_id"].tolist()
    rid_list = restaurants["restaurant_id"].tolist()

    if user_id not in uid_list:
        return jsonify({"error": f"user_id {user_id} not found"}), 404
    if restaurant_id not in rid_list:
        return jsonify({"error": f"restaurant_id {restaurant_id} not found"}), 404

    i = uid_list.index(user_id)
    j = rid_list.index(restaurant_id)
    R[i, j] = float(rating_value)

    return jsonify({"status": "ok"})


@app.post("/recommend")
def recommend():
    body        = request.get_json(force=True)
    user_id     = body.get("user_id")
    preferences = body.get("preferences", {})
    top_n       = int(body.get("top_n", 5))

    recs = get_recommendations(
        user_id=user_id,
        preferences=preferences,
        R=R,
        users=users,
        restaurants=restaurants,
        top_n=top_n,
    )

    # Rename final_score → predicted_score to match the frontend template
    cuisine_filter = (preferences.get("cuisine") or "").strip().lower()
    results = []
    for r in recs:
        if cuisine_filter and r["cuisine"].lower() != cuisine_filter:
            continue
        r["predicted_score"] = r.pop("final_score")
        results.append(r)

    return jsonify(results)


# ── Serve the frontend ────────────────────────────────────────────────────────

@app.get("/")
def index():
    from flask import render_template
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
