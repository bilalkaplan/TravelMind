from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT_PATH = Path("data/processed/balanced_llm_comment_predictions.csv")
REPORT_PATH = Path("data/processed/balanced_llm_comment_evaluation_report.txt")


def main():
    if not INPUT_PATH.exists():
        print("Dosya bulunamadı:", INPUT_PATH)
        print("Önce şunu çalıştır:")
        print("python src\\llm_balanced_comment_analyzer.py")
        return

    df = pd.read_csv(INPUT_PATH)

    required_columns = ["manual_sentiment_label", "llm_sentiment"]

    for column in required_columns:
        if column not in df.columns:
            print(f"Eksik kolon: {column}")
            return

    df["manual_sentiment_label"] = (
        df["manual_sentiment_label"].fillna("").astype(str).str.strip().str.lower()
    )

    df["llm_sentiment"] = (
        df["llm_sentiment"].fillna("").astype(str).str.strip().str.lower()
    )

    eval_df = df[
        (df["manual_sentiment_label"] != "") & (df["llm_sentiment"] != "")
    ].copy()

    if eval_df.empty:
        print("Değerlendirilecek etiketli veri yok.")
        return

    y_true = eval_df["manual_sentiment_label"]
    y_pred = eval_df["llm_sentiment"]

    labels = ["positive", "negative", "neutral", "unclear"]

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    manual_counts = eval_df["manual_sentiment_label"].value_counts().to_string()
    llm_counts = eval_df["llm_sentiment"].value_counts().to_string()

    wrong_df = eval_df[
        eval_df["manual_sentiment_label"] != eval_df["llm_sentiment"]
    ].copy()

    if wrong_df.empty:
        wrong_examples = "No wrong predictions."
    else:
        columns = [
            "sample_id",
            "room_comment",
            "manual_sentiment_label",
            "llm_sentiment",
            "sarcasm_or_irony",
            "llm_sarcasm_or_irony",
            "llm_confidence",
            "llm_reason",
        ]

        existing_columns = [col for col in columns if col in wrong_df.columns]
        wrong_examples = wrong_df[existing_columns].to_string(index=False)

    output = f"""
TravelMind RAG - Balanced LLM Comment Understanding Evaluation

Total evaluated samples: {len(eval_df)}

Manual Label Distribution:
{manual_counts}

LLM Prediction Distribution:
{llm_counts}

Accuracy:
{accuracy:.4f}

Classification Report:
{report}

Confusion Matrix

Labels order:
{labels}

{matrix}

Wrong Predictions:
{wrong_examples}

Interpretation:
- This evaluation uses a more balanced test set than the first 10-sample test.
- Positive, negative and neutral samples are selected from the dataset.
- Unclear/sarcastic examples are added as a challenge set to test whether the LLM avoids overconfident interpretation.
- Accuracy shows the overall correctness.
- Macro F1 is important because it treats each class equally.
- Weighted F1 accounts for class imbalance.
- Sarcastic, ironic, mixed or vague comments should usually be labeled as unclear.
"""

    REPORT_PATH.write_text(output, encoding="utf-8")

    print(output)
    print("Rapor oluşturuldu:", REPORT_PATH)


if __name__ == "__main__":
    main()
