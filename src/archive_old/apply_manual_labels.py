from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/llm_comment_predictions.csv")
OUTPUT_PATH = Path("data/processed/llm_comment_predictions_labeled.csv")

MANUAL_LABELS = {
    4608: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "Kısa ve eksik görünse de 'Beautiful room' ifadesi olumlu.",
    },
    5033: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "'room was clean' doğrudan olumlu temizlik ifadesi.",
    },
    2592: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "'Excellent hotel' açık olumlu ifade.",
    },
    3151: {
        "manual_sentiment_label": "neutral",
        "sarcasm_or_irony": "no",
        "notes": "Oda yapısı tarif ediliyor; güçlü olumlu/olumsuz duygu yok.",
    },
    5776: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "Yardım ve düzenleme için memnuniyet ifade edilmiş.",
    },
    4396: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "'room was clean' olumlu.",
    },
    6384: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "'Fantastic location', 'pleasant staff' ve 'well maintained' olumlu.",
    },
    4398: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "'great room' açık olumlu.",
    },
    5513: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "'I wish it could be longer' memnuniyet ve daha uzun kalma isteği gösteriyor.",
    },
    4797: {
        "manual_sentiment_label": "positive",
        "sarcasm_or_irony": "no",
        "notes": "'Great view from our room' açık olumlu.",
    },
}


def main():
    if not INPUT_PATH.exists():
        print("Dosya bulunamadı:", INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    # Boş kolonları pandas float64 okuyabildiği için metin tipine çeviriyoruz.
    for column in ["manual_sentiment_label", "sarcasm_or_irony", "notes"]:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").astype("object")

    df["hotel_id"] = pd.to_numeric(df["hotel_id"], errors="coerce").astype("Int64")

    for hotel_id, labels in MANUAL_LABELS.items():
        mask = df["hotel_id"] == hotel_id

        if not mask.any():
            print(f"Uyarı: hotel_id bulunamadı: {hotel_id}")
            continue

        df.loc[mask, "manual_sentiment_label"] = labels["manual_sentiment_label"]
        df.loc[mask, "sarcasm_or_irony"] = labels["sarcasm_or_irony"]
        df.loc[mask, "notes"] = labels["notes"]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Manuel etiketler yazıldı.")
    print("Yeni dosya:", OUTPUT_PATH)

    print("\nÖzet:")
    print(
        df[
            [
                "hotel_id",
                "hotel_name",
                "manual_sentiment_label",
                "llm_sentiment",
                "sarcasm_or_irony",
                "llm_sarcasm_or_irony",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
