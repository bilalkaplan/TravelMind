from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT_PATH = Path("data/processed/llm_comment_predictions_labeled.csv")
REPORT_PATH = Path("data/processed/llm_comment_evaluation_report.txt")


def main():
    if not INPUT_PATH.exists():
        print("Dosya bulunamadı:", INPUT_PATH)
        print("Önce şunu çalıştır:")
        print("python src\\llm_comment_analyzer.py")
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
        print("Henüz manuel etiket yok.")
        print("Şu dosyayı aç:")
        print(INPUT_PATH)
        print("manual_sentiment_label kolonunu doldur.")
        print("Kullanılabilecek etiketler:")
        print("positive, negative, neutral, unclear")
        return

    y_true = eval_df["manual_sentiment_label"]
    y_pred = eval_df["llm_sentiment"]

    labels = ["positive", "negative", "neutral", "unclear"]

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    output = f"""
TravelMind RAG - LLM Comment Understanding Evaluation

Total evaluated samples: {len(eval_df)}

Accuracy:
{accuracy:.4f}

Classification Report:
{report}

Confusion Matrix
Labels order:
{labels}

{matrix}

Interpretation:
- Accuracy shows the overall ratio of correct LLM sentiment labels.
- Macro F1 treats each class equally.
- Weighted F1 accounts for class imbalance.
- Sarcastic, ironic, mixed or vague comments should usually be labeled as unclear.
"""

    REPORT_PATH.write_text(output, encoding="utf-8")

    print(output)
    print("Rapor oluşturuldu:", REPORT_PATH)


if __name__ == "__main__":
    main()
