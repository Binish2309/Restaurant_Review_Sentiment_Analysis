"""
Dataset Generator for Restaurant Review Sentiment Analysis
Generates a realistic synthetic dataset in the absence of a pre-downloaded Kaggle CSV.
Run this once: python dataset/generate_dataset.py
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

RESTAURANTS = [
    "The Golden Fork", "Spice Garden", "La Bella Italia", "Sushi Paradise",
    "Burger Barn", "The Cozy Diner", "Mediterranean Breeze", "Dragon Palace",
    "Taco Fiesta", "The Steakhouse", "Green Leaf Bistro", "Pho Saigon",
    "Pizza Palace", "The BBQ Pit", "Sea Breeze Seafood"
]

POSITIVE_REVIEWS = [
    "Absolutely loved the food here! The pasta was cooked to perfection.",
    "Amazing service and delicious food. Will definitely come back!",
    "The best burger I have ever had. Fresh ingredients and great taste.",
    "Wonderful dining experience. Staff was very friendly and attentive.",
    "The ambiance was fantastic and the sushi was incredibly fresh.",
    "Highly recommend this place. Every dish was a masterpiece.",
    "Great value for money. Huge portions and outstanding flavors.",
    "The chef really knows his craft. Everything was seasoned perfectly.",
    "Best restaurant in town! Our whole family enjoyed the meal.",
    "Excellent food quality and super fast service. Loved the desserts too!",
    "We celebrated our anniversary here and it was magical. Perfect evening.",
    "The steak was juicy, tender, and cooked exactly as requested. Bravo!",
    "Fresh seafood, great wine selection, and a beautiful ocean view.",
    "The tacos here are out of this world. Authentic and flavorful!",
    "Incredible breakfast menu. The pancakes were fluffy and divine.",
    "Loved the vegetarian options. Finally a restaurant that caters to all!",
    "The soup was rich, creamy, and warming. Just what I needed.",
    "Service was prompt and the staff remembered our preferences. Impressive!",
    "Clean, modern, and sophisticated. The food matched the decor.",
    "Generous portions, reasonable prices, and top-notch quality. Five stars!",
]

NEGATIVE_REVIEWS = [
    "Terrible experience. The food was cold and the service was rude.",
    "Waited over an hour for our food. Completely unacceptable.",
    "The pizza was soggy and tasteless. Would not recommend.",
    "Overpriced for what you get. The portions were tiny.",
    "The staff ignored us for 20 minutes. Very disappointing.",
    "Found a hair in my food. Extremely unhygienic.",
    "The restaurant was dirty and the tables were sticky.",
    "Food was undercooked and gave me a stomachache. Never coming back.",
    "Wrong order delivered twice. Kitchen needs serious management.",
    "The noise level was unbearable. Could not hold a conversation.",
    "Rude manager refused to acknowledge our complaint. Worst experience ever.",
    "The seafood smelled off. Lucky I did not get food poisoning.",
    "Waited 40 minutes and the food was mediocre at best.",
    "Menu items listed were unavailable. Very misleading and frustrating.",
    "The restrooms were filthy. If they do not maintain this, imagine the kitchen!",
    "Stale bread and cold soup. This place has really gone downhill.",
    "Overcooked pasta, bland sauce, and zero customer care.",
    "Parking lot full and no alternative given. Staff seemed indifferent.",
    "They messed up three orders at our table. Completely disorganized.",
    "Not worth the hype. The food was average and the bill was shocking.",
]

NEUTRAL_REVIEWS = [
    "The food was okay. Nothing special but not bad either.",
    "Average experience overall. The service was adequate.",
    "Decent place for a quick bite. Nothing to write home about.",
    "The food was fine. Probably would not make a special trip here.",
    "It was alright. The location is convenient which helps.",
    "Standard restaurant experience. Met basic expectations.",
    "The menu has a good variety but the execution was average.",
    "Service was okay but could be more attentive.",
    "Food was edible and reasonably priced. Nothing extraordinary.",
    "Visited for lunch. The meal was satisfying but unremarkable.",
    "Typical chain restaurant food. Consistent but uninspiring.",
    "Not bad, not great. Exactly what you would expect from this type of place.",
    "The portions were decent and the prices were fair. A regular spot.",
    "Neither impressed nor disappointed. It served its purpose.",
    "Would visit again if in the area but would not go out of my way.",
]

records = []
start_date = pd.Timestamp("2022-01-01")
end_date = pd.Timestamp("2024-12-31")

for i in range(1000):
    restaurant = random.choice(RESTAURANTS)
    sentiment_choice = random.choices(
        ["Positive", "Negative", "Neutral"],
        weights=[0.55, 0.30, 0.15],
        k=1
    )[0]

    if sentiment_choice == "Positive":
        review = random.choice(POSITIVE_REVIEWS)
        rating = random.choices([4, 5], weights=[0.3, 0.7])[0]
    elif sentiment_choice == "Negative":
        review = random.choice(NEGATIVE_REVIEWS)
        rating = random.choices([1, 2], weights=[0.5, 0.5])[0]
    else:
        review = random.choice(NEUTRAL_REVIEWS)
        rating = 3

    date = start_date + (end_date - start_date) * random.random()
    reviewer_id = f"reviewer_{random.randint(1000, 9999)}"

    records.append({
        "reviewer_id": reviewer_id,
        "restaurant_name": restaurant,
        "review": review,
        "rating": rating,
        "date": date.strftime("%Y-%m-%d"),
        "true_sentiment": sentiment_choice
    })

df = pd.DataFrame(records)
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/restaurant_reviews.csv", index=False)
print(f"Dataset generated: {len(df)} reviews saved to dataset/restaurant_reviews.csv")
print(df.head())
print(df["true_sentiment"].value_counts())
