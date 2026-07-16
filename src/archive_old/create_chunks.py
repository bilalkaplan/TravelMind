import json
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/reviews_clean.csv")
OUTPUT_PATH = Path("data/processed/review_chunks.jsonl")


def main():
    df = pd.read_csv(INPUT_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            chunk = {
                "chunk_id": f"review_{int(row['review_id'])}",
                "text": row["review"],
                "metadata": {
                    "review_id": int(row["review_id"]),
                    "rating": int(row["rating"]),
                    "sentiment": row["sentiment"],
                    "source": "Trip Advisor Hotel Reviews",
                    "dataset": "andrewmvd/trip-advisor-hotel-reviews",
                },
            }

            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Chunk dosyası oluşturuldu:", OUTPUT_PATH)
    print("Toplam chunk sayısı:", len(df))

    print("\nİlk örnek chunk:")
    print(json.dumps(chunk, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
