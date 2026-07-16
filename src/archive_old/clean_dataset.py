import re
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/tripadvisor_hotel_reviews.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "reviews_clean.csv"


def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def rating_to_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    print("İlk veri boyutu:", df.shape)

    df = df.rename(columns={"Review": "review", "Rating": "rating"})

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review"] = df["review"].apply(clean_text)

    df = df.dropna(subset=["review", "rating"])
    df = df[df["review"].str.len() >= 30]
    df = df[df["rating"].between(1, 5)]

    df = df.drop_duplicates(subset=["review"])

    df["sentiment"] = df["rating"].apply(rating_to_sentiment)

    df = df.reset_index(drop=True)
    df.insert(0, "review_id", df.index + 1)

    df = df[["review_id", "review", "rating", "sentiment"]]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Temizlenmiş veri boyutu:", df.shape)
    print("Kaydedilen dosya:", OUTPUT_PATH)

    print("\nRating dağılımı:")
    print(df["rating"].value_counts().sort_index())

    print("\nSentiment dağılımı:")
    print(df["sentiment"].value_counts())

    print("\nİlk 5 temiz kayıt:")
    print(df.head())


if __name__ == "__main__":
    main()
