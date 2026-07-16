from pathlib import Path
import pandas as pd

REVIEWS_PATH = Path("data/processed/reviews_clean.csv")
OUTPUT_PATH = Path("data/processed/balanced_comment_eval_set.csv")

RANDOM_STATE = 42

TARGET_COUNTS = {"positive": 10, "negative": 10, "neutral": 10, "unclear": 5}

UNCLEAR_CHALLENGE_EXAMPLES = [
    {
        "room_comment": 'Great, another sleepless night in a "quiet" hotel.',
        "manual_sentiment_label": "negative",
        "sarcasm_or_irony": "yes",
        "notes": "Sarcastic comment with a clear negative meaning.",
    },
    {
        "room_comment": "The room was fine, I guess.",
        "manual_sentiment_label": "neutral",
        "sarcasm_or_irony": "unclear",
        "notes": "Lukewarm and vague, but closer to neutral than clearly positive or negative.",
    },
    {
        "room_comment": "Not bad, not great either.",
        "manual_sentiment_label": "neutral",
        "sarcasm_or_irony": "no",
        "notes": "Mixed but understandable neutral comment.",
    },
    {
        "room_comment": "If you enjoy listening to elevators all night, this is perfect.",
        "manual_sentiment_label": "negative",
        "sarcasm_or_irony": "yes",
        "notes": "Sarcastic comment with a clear negative meaning about noise.",
    },
    {
        "room_comment": "The hotel was... interesting.",
        "manual_sentiment_label": "unclear",
        "sarcasm_or_irony": "unclear",
        "notes": "Ambiguous wording; sentiment is not clear.",
    },
]


def clean_text(text):
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def load_review_data():
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {REVIEWS_PATH}")

    df = pd.read_csv(REVIEWS_PATH)

    if "review" not in df.columns:
        raise ValueError("reviews_clean.csv içinde 'review' kolonu yok.")

    if "sentiment" not in df.columns:
        if "rating" not in df.columns:
            raise ValueError(
                "reviews_clean.csv içinde ne 'sentiment' ne de 'rating' kolonu var."
            )

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        def rating_to_sentiment(rating):
            if rating >= 4:
                return "positive"
            if rating <= 2:
                return "negative"
            return "neutral"

        df["sentiment"] = df["rating"].apply(rating_to_sentiment)

    df["review"] = df["review"].fillna("").astype(str).apply(clean_text)
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.lower().str.strip()

    df = df[df["review"].str.len() >= 40]
    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]

    return df


def sample_reviews(df, label, count):
    subset = df[df["sentiment"] == label].copy()

    if subset.empty:
        print(f"Uyarı: {label} için hiç örnek bulunamadı.")
        return pd.DataFrame()

    sample_count = min(count, len(subset))

    sampled = subset.sample(n=sample_count, random_state=RANDOM_STATE).copy()

    output = pd.DataFrame()

    output["sample_id"] = [f"{label}_{i + 1}" for i in range(len(sampled))]
    output["dataset_source"] = "tripadvisor_reviews_rating_based"
    output["hotel_id"] = ""
    output["hotel_name"] = ""
    output["location"] = ""
    output["hotel_rating"] = ""
    output["room_score"] = ""
    output["review_count"] = ""
    output["room_type"] = ""
    output["bed_type"] = ""
    output["room_comment"] = sampled["review"].values
    output["source"] = "reviews_clean.csv"
    output["manual_sentiment_label"] = label
    output["sarcasm_or_irony"] = "no"
    output["notes"] = "Silver label derived from dataset rating/sentiment."

    return output


def create_unclear_examples():
    rows = []

    for i, item in enumerate(UNCLEAR_CHALLENGE_EXAMPLES, start=1):
        rows.append(
            {
                "sample_id": f"unclear_{i}",
                "dataset_source": "challenge_sarcasm_unclear_set",
                "hotel_id": "",
                "hotel_name": "",
                "location": "",
                "hotel_rating": "",
                "room_score": "",
                "review_count": "",
                "room_type": "",
                "bed_type": "",
                "room_comment": item["room_comment"],
                "source": "challenge_set",
                "manual_sentiment_label": item["manual_sentiment_label"],
                "sarcasm_or_irony": item["sarcasm_or_irony"],
                "notes": item["notes"],
            }
        )

    return pd.DataFrame(rows)


def main():
    df = load_review_data()

    parts = []

    parts.append(sample_reviews(df, "positive", TARGET_COUNTS["positive"]))
    parts.append(sample_reviews(df, "negative", TARGET_COUNTS["negative"]))
    parts.append(sample_reviews(df, "neutral", TARGET_COUNTS["neutral"]))
    parts.append(create_unclear_examples())

    final_df = pd.concat(parts, ignore_index=True)

    final_df = final_df[
        [
            "sample_id",
            "dataset_source",
            "hotel_id",
            "hotel_name",
            "location",
            "hotel_rating",
            "room_score",
            "review_count",
            "room_type",
            "bed_type",
            "room_comment",
            "source",
            "manual_sentiment_label",
            "sarcasm_or_irony",
            "notes",
        ]
    ]

    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Dengeli evaluation seti oluşturuldu:")
    print(OUTPUT_PATH)

    print("\nSınıf dağılımı:")
    print(final_df["manual_sentiment_label"].value_counts())

    print("\nSarcasm/Irony dağılımı:")
    print(final_df["sarcasm_or_irony"].value_counts())

    print("\nToplam örnek sayısı:", len(final_df))


if __name__ == "__main__":
    main()
