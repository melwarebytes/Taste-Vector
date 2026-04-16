# TasteVector

TasteVector is a personalized restaurant recommendation system built as a Linear Algebra applied project. It uses SVD collaborative filtering, eigendecomposition, Gram-Schmidt projections, and PageRank-style ranking to recommend restaurants based on user preferences and rating history.

## Project Structure

```text
data/
  ratings.csv         # User–restaurant ratings (user_id, restaurant_id, rating 1–5)
  restaurants.csv     # Restaurant metadata (id, name, cuisine, price, spice, distance_km, veg_friendly)
  users.csv           # User profiles (id, name, preferred_cuisine, max_price, spice_tolerance, max_distance)

src/
  data_loader.py      # CSV ingestion and validation
  matrix_builder.py   # Builds the rating matrix R
  recommender.py      # Main pipeline — orchestrates SVD, PageRank, and cold-start
  svd_recommender.py  # SVD collaborative filtering
  eigen_decomp.py     # Covariance matrix and top-k eigenvectors
  pagerank_ranker.py  # PageRank-style global importance scores
  gram_schmidt.py     # Gram-Schmidt orthogonalization
  projection.py       # Least-squares projection (cold-start)
  subspace_analysis.py
  similarity.py
  gaussian_elimination.py

app/
  api.py              # Flask API entry point
  templates/
    index.html        # Frontend UI
  static/
    style.css

tests/                # pytest test suite
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the App

Start the Flask backend:

```bash
flask --app app/api.py run
```

Then open `http://127.0.0.1:5000` in your browser.

The UI lets you set preferences (max price, spice tolerance, max distance, cuisine), optionally enter a user ID to use your rating history, and get ranked restaurant recommendations.

## Running Tests

```bash
pytest tests/ -v
```

## How It Works

**Existing users** — SVD collaborative filtering predicts ratings, blended with a PageRank global importance score.

**New / anonymous users** (cold-start) — Preference vector is projected onto the top eigenvectors of the restaurant feature covariance matrix, then blended with PageRank.

Final score: `0.60 × SVD + 0.25 × PageRank + 0.15 × projection (cold-start only)`

Restaurants exceeding your max price or max distance are filtered out entirely.

## Quick Smoke Test

```bash
python src/recommender.py
```

Prints sample recommendations for an existing user and a new user against the data in `data/`.
