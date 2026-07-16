import json
import re
from pathlib import Path

import pandas as pd
import openai  # type: ignore
from foundry_local_sdk import Configuration, FoundryLocalManager  # type: ignore

MODEL_ALIAS = "phi-4-mini"

INPUT_PATH = Path("data/processed/manual_comment_eval_sample.csv")
OUTPUT_PATH = Path("data/processed/llm_comment_predictions.csv")


SYSTEM_PROMPT = """
You are an evaluator for hotel room comments.

Your task is to classify the comment using only the given comment text.

Return ONLY valid JSON with exactly these fields:
{
  "sentiment": "positive | negative | neutral | unclear",
  "sarcasm_or_irony": "yes | no | unclear",
  "confidence": "high | medium | low",
  "reason": "short explanation"
}

Rules:
- Do not invent facts.
- Do not use outside knowledge.
- If the comment is sarcastic, ironic, mixed, or ambiguous, use "unclear" when necessary.
- If the comment contains both positive and negative points, use "neutral" unless one side is clearly dominant.
- If the comment is too short or vague, use "unclear".
- The reason must be short.
"""


def extract_json(text):
    text = str(text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "sentiment": "unclear",
        "sarcasm_or_irony": "unclear",
        "confidence": "low",
        "reason": "Model response could not be parsed as JSON.",
    }


def normalize_label(value, allowed_values, default_value):
    value = str(value).strip().lower()

    if value in allowed_values:
        return value

    return default_value


def initialize_foundry_client():
    print("Foundry Local başlatılıyor...")

    config = Configuration(app_name="travelmind_rag")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    print("Execution provider kontrolü yapılıyor...")
    manager.download_and_register_eps()

    print(f"Model hazırlanıyor: {MODEL_ALIAS}")
    model = manager.catalog.get_model(MODEL_ALIAS)

    model.download(
        lambda progress: print(
            f"\rModel kontrol/indirme: {progress:.2f}%", end="", flush=True
        )
    )

    print()

    model.load()
    print("Model yüklendi.")

    manager.start_web_service()

    base_url = f"{manager.urls[0]}/v1"

    client = openai.OpenAI(base_url=base_url, api_key="none")

    return manager, model, client


def analyze_comment(client, model_id, comment):
    user_prompt = f"""
Hotel room comment:
{comment}
"""

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=220,
    )

    content = response.choices[0].message.content
    parsed = extract_json(content)

    sentiment = normalize_label(
        parsed.get("sentiment", ""),
        {"positive", "negative", "neutral", "unclear"},
        "unclear",
    )

    sarcasm_or_irony = normalize_label(
        parsed.get("sarcasm_or_irony", ""), {"yes", "no", "unclear"}, "unclear"
    )

    confidence = normalize_label(
        parsed.get("confidence", ""), {"high", "medium", "low"}, "low"
    )

    reason = str(parsed.get("reason", "")).strip()

    return {
        "llm_sentiment": sentiment,
        "llm_sarcasm_or_irony": sarcasm_or_irony,
        "llm_confidence": confidence,
        "llm_reason": reason,
        "raw_llm_response": content,
    }


def main():
    if not INPUT_PATH.exists():
        print("Dosya bulunamadı:", INPUT_PATH)
        print("Önce şunu çalıştır:")
        print("python src\\audit_quality_and_evaluation_plan.py")
        return

    df = pd.read_csv(INPUT_PATH)

    if "room_comment" not in df.columns:
        print("manual_comment_eval_sample.csv içinde room_comment kolonu yok.")
        return

    df = df.copy()

    df["room_comment"] = df["room_comment"].fillna("").astype(str)
    df = df[df["room_comment"].str.strip() != ""]

    if df.empty:
        print("Analiz edilecek room_comment bulunamadı.")
        return

    print(f"Toplam yorum sayısı: {len(df)}")
    limit_text = input("Kaç yorum analiz edilsin? İlk deneme için 10 yaz: ").strip()

    try:
        limit = int(limit_text)
    except ValueError:
        limit = 10

    limit = max(1, min(limit, len(df)))
    df = df.head(limit).copy()

    manager = None
    model = None

    try:
        manager, model, client = initialize_foundry_client()

        results = []

        for index, row in df.iterrows():
            comment = row["room_comment"]

            print("\n" + "-" * 80)
            print(f"Yorum analiz ediliyor: {len(results) + 1} / {len(df)}")
            print(comment[:300])

            analysis = analyze_comment(
                client=client, model_id=model.id, comment=comment
            )

            output_row = row.to_dict()
            output_row.update(analysis)
            results.append(output_row)

            print("LLM sentiment:", analysis["llm_sentiment"])
            print("Sarcasm/Irony:", analysis["llm_sarcasm_or_irony"])
            print("Confidence:", analysis["llm_confidence"])
            print("Reason:", analysis["llm_reason"])

        output_df = pd.DataFrame(results)
        output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

        print("\n" + "=" * 80)
        print("LLM yorum analizi tamamlandı.")
        print("Çıktı dosyası:", OUTPUT_PATH)

    finally:
        if model is not None:
            try:
                model.unload()
                print("Model unload edildi.")
            except Exception as error:
                print("Model unload sırasında hata:", error)

        if manager is not None:
            try:
                manager.stop_web_service()
                print("Foundry web service durduruldu.")
            except Exception as error:
                print("Web service durdurulurken hata:", error)


if __name__ == "__main__":
    main()
