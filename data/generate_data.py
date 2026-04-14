"""
Generates synthetic data for TasteVector into restaurants.csv, users.csv, ratings.csv
Matches the schemas provided by Person 1.
"""
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

CUISINES = ["Italian", "Indian", "Chinese", "Mexican", "Japanese", "American", "Thai", "Mediterranean"]

# --- restaurants.csv ---
# restaurant_id, name, cuisine, price, spice, distance_km, veg_friendly
restaurants = []
names = [
    "Spice Garden", "Bella Italia", "Dragon Palace", "Taco Fiesta", "Sakura House",
    "Burger Barn", "Thai Orchid", "Olive Grove", "Curry Leaf", "Pasta Prima",
    "Wok & Roll", "El Rancho", "Sushi Bay", "American Grill", "Pad Thai Place",
    "Mezze Corner", "Naan Stop", "Pizza Pronto", "Peking Duck", "La Cantina"
]
for i, name in enumerate(names):
    restaurants.append({
        "restaurant_id": f"R{i+1:02d}",
        "name": name,
        "cuisine": CUISINES[i % len(CUISINES)],
        "price": round(random.uniform(100, 800)),       # price in INR
        "spice": round(random.uniform(1, 5), 1),         # 1=mild, 5=very spicy
        "distance_km": round(random.uniform(0.5, 15.0), 1),
        "veg_friendly": random.choice([True, False])
    })

df_restaurants = pd.DataFrame(restaurants)
df_restaurants.to_csv("/home/claude/tastevector/data/restaurants.csv", index=False)

# --- users.csv ---
# user_id, name, preferred_cuisine, max_price, spice_tolerance, max_distance
user_names = [
    "Aarav", "Meera", "Rahul", "Priya", "Karan",
    "Sneha", "Vikram", "Ananya", "Rohan", "Divya",
    "Arjun", "Pooja", "Nikhil", "Shruti", "Aditya"
]
users = []
for i, uname in enumerate(user_names):
    users.append({
        "user_id": f"U{i+1:02d}",
        "name": uname,
        "preferred_cuisine": random.choice(CUISINES),
        "max_price": random.choice([200, 300, 500, 700, 1000]),
        "spice_tolerance": round(random.uniform(1, 5), 1),
        "max_distance": random.choice([3, 5, 8, 10, 15])
    })

df_users = pd.DataFrame(users)
df_users.to_csv("/home/claude/tastevector/data/users.csv", index=False)

# --- ratings.csv ---
# user_id, restaurant_id, rating (sparse — not every user rates every restaurant)
ratings = []
for u in users:
    # Each user rates ~40–70% of restaurants
    rated = random.sample(restaurants, k=random.randint(8, 14))
    for r in rated:
        ratings.append({
            "user_id": u["user_id"],
            "restaurant_id": r["restaurant_id"],
            "rating": round(random.uniform(1, 5), 1)
        })

df_ratings = pd.DataFrame(ratings)
df_ratings.to_csv("/home/claude/tastevector/data/ratings.csv", index=False)

print(f"Generated: {len(restaurants)} restaurants, {len(users)} users, {len(ratings)} ratings")
print(df_restaurants.head(3))
print(df_users.head(3))
print(df_ratings.head(5))
