import os
import sys
import typing
import json
import re
import subprocess
import gc
import torch

def get_truncated_history(history):
    return [{'role': msg['role'], 'content': msg['content'][:300] + '...' if len(msg['content']) > 300 else msg['content']} for msg in history]

HOTEL_METADATA_CACHE = None

def load_hotel_metadata():
    global HOTEL_METADATA_CACHE
    if HOTEL_METADATA_CACHE is None:
        try:
            with open('data/cmu_hotel_metadata.json', 'r', encoding='utf-8') as f:
                HOTEL_METADATA_CACHE = json.load(f)
        except Exception as e:
            HOTEL_METADATA_CACHE = {}
    return HOTEL_METADATA_CACHE

from openai import OpenAI, BadRequestError  # type: ignore
from rich.console import Console
from rich.markdown import Markdown

from cmu_retrieve import search
from cmu_recommend_hotels import (
    calculate_recommendation_score,
    extract_total_review_count,
    format_avg,
    build_strengths,
    build_cautions,
)
from language_utils import (
    detect_language,
    language_name,
    no_result_message,
    empty_query_message,
    goodbye_message,
    input_prompt,
    initial_welcome_message,
    unsupported_location_message,
)

# Reconfigure stdout for Windows emoji support
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

TOP_K_RETRIEVAL = 10
TOP_K_FOR_LLM = 3
DEBUG = False


def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clear_python_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Python GPU cache temizlendi.")


def get_foundry_base_url():
    if DEBUG:
        print("Foundry Local endpoint aranıyor...")

    result = subprocess.run(
        ["foundry", "service", "status"], capture_output=True, text=True, check=False
    )

    output = result.stdout + "\n" + result.stderr

    match = re.search(r"http://127\.0\.0\.1:\d+", output)
    if not match:
        match = re.search(r"http://localhost:\d+", output)

    if not match:
        print("\nFoundry service status çıktısı:")
        print(output)
        raise RuntimeError("Foundry endpoint bulunamadı.")

    endpoint = match.group(0)

    if not endpoint.endswith("/v1"):
        endpoint = endpoint + "/v1"

    if DEBUG:
        print("Foundry endpoint:", endpoint)
    return endpoint


def get_available_model_id(client):
    if DEBUG:
        print("Foundry endpoint üzerindeki modeller kontrol ediliyor...")

    models = client.models.list()
    model_ids = [model.id for model in models.data]

    if DEBUG:
        print("Bulunan modeller:")
        for model_id in model_ids:
            print("-", model_id)

    if not model_ids:
        raise RuntimeError("Foundry endpoint üzerinde model bulunamadı.")

    for model_id in model_ids:
        if "phi" in model_id.lower():
            if DEBUG:
                print("Kullanılacak model:", model_id)
            return model_id

    selected_model = model_ids[0]
    if DEBUG:
        print("Phi modeli bulunamadı, ilk model kullanılacak:", selected_model)
    return selected_model


def build_hotel_context(result, index):
    metadata = result["metadata"]
    text = result["text"]
    scoring = calculate_recommendation_score(result)

    hotel_name = metadata.get("hotel_name", "")
    location = metadata.get("location", "")
    hotel_class = metadata.get("hotel_class", "")
    total_review_count = extract_total_review_count(metadata, text)

    strengths = build_strengths(result, scoring)
    cautions = build_cautions(result)

    normalized_text = str(text).lower()
    has_single_room = "single room" in normalized_text or "tek kişilik" in normalized_text
    has_double_room = "double room" in normalized_text or "çift kişilik" in normalized_text
    
    room_info = "Unknown / Not explicitly guaranteed in this dataset chunk."
    if has_single_room and has_double_room:
        room_info = "Mentioned single and double rooms."
    elif has_single_room:
        room_info = "Mentioned single rooms."
    elif has_double_room:
        room_info = "Mentioned double rooms."
        
    # Inject exact address from OSM Metadata cache
    metadata_cache = load_hotel_metadata()
    exact_address = "Not Available"
    if metadata_cache:
        hotel_key = f"{hotel_name}::{location}"
        if hotel_key in metadata_cache:
            osm_data = metadata_cache[hotel_key].get("osm_data")
            if osm_data and "address" in osm_data:
                exact_address = osm_data["address"]

    context = f"""
Hotel Card {index}:
- Hotel name: {hotel_name}
- Location: {location}
- Exact Address: {exact_address}
- TravelMind suitability score: {scoring["score"]}/100
- Overall rating: {format_avg(scoring["overall_avg"])}
- Cleanliness rating: {format_avg(scoring["cleanliness_avg"])}
- Location rating: {format_avg(scoring["location_avg"])}
- Service rating: {format_avg(scoring["service_avg"])}
- Rooms rating: {format_avg(scoring["rooms_avg"])}
- Total review count: {total_review_count}
- Strengths: {', '.join(strengths) if strengths else 'None specific'}
- Cautions: {', '.join(cautions) if cautions else 'None specific'}
- Room types explicitly mentioned in chunk: {room_info}
- Score Components: {str(scoring.get("components", {}))}
- Score Weights: {str(scoring.get("weights", {}))}
"""

    return context.strip()



def get_style_instruction(language):
    if language == "Turkish":
        return """
SİZİN İÇİN KESİN KURAL: Aşağıdaki "Örnek Çıktı" (Example Output) şablonunu BİREBİR KOPYALAYIN. Sadece köşeli parantez içindeki yerleri "Retrieved hotel evidence" kısmındaki gerçek verilerle doldurun. Asla kendi yorumunuzu veya talimatları ekrana basmayın!

Örnek Çıktı:
Size yardımcı olmaktan büyük mutluluk duyarım. Belirttiğiniz tercihlere en uygun otelleri özenle seçtim:

### [Otel Adı 1]
- **Adres:** [Tam Adres]
- **TravelMind Skoru:** [Skor] / 100
- **Genel:** [Puan] | **Temizlik:** [Puan] | **Konum:** [Puan]
- **Öne Çıkanlar:** [Temizliği çok iyi vs.]
- **Dikkat Edilmesi Gerekenler:** [Bazı yorumlar karışık vs.]

### [Otel Adı 2]
- **Adres:** [Tam Adres]
- **TravelMind Skoru:** [Skor] / 100
- **Genel:** [Puan] | **Temizlik:** [Puan] | **Konum:** [Puan]
- **Öne Çıkanlar:** [Konumu merkeze yakın vs.]
- **Dikkat Edilmesi Gerekenler:** [Yok]

*Not: Özel olarak aradığınız [Tek Kişilik Oda vb.] detaylar sistemimizde bulunmuyor olabilir, ancak yukarıdaki seçenekler konforlu bir konaklama için idealdir.*

Başka bir sorunuz veya farklı bir konum tercihiniz olursa lütfen bana bildirin.
""".strip()
    else:
        return """
STRICT RULE: You MUST perfectly mirror the "Example Output" template below. Only fill in the bracketed placeholders using the provided "Retrieved hotel evidence". Never output instructions or meta-text!

Example Output:
It is my pleasure to assist you. I have carefully selected the best hotels that match your preferences:

### [Hotel Name 1]
- **Address:** [Exact Address]
- **TravelMind Score:** [Score] / 100
- **Overall:** [Score] | **Cleanliness:** [Score] | **Location:** [Score]
- **Highlights:** [Great cleanliness etc.]
- **Cautions:** [Mixed reviews etc.]

### [Hotel Name 2]
- **Address:** [Exact Address]
- **TravelMind Score:** [Score] / 100
- **Overall:** [Score] | **Cleanliness:** [Score] | **Location:** [Score]
- **Highlights:** [Excellent location etc.]
- **Cautions:** [None]

*Note: Specific details like [Single Room etc.] might not be available in our records, but the options above offer great comfort.*

Please let me know if you have any other questions or need further assistance.
""".strip()

def build_prompt(query, language, hotel_context):
    style_instruction = get_style_instruction(language)

    return f"""
User query:
{query}

The answer must be written in:
{language}

Important language rules:
- The answer language is strictly {language}. Answer ONLY in {language}.
- Do not switch languages.

Retrieved hotel evidence:
{hotel_context}

Write the final TravelMind answer for the user.

Strict rules:
- Answer ONLY in {language}.
- Use ONLY the hotel options provided in the evidence. DO NOT invent extra hotels.
- If the retrieved hotel evidence does not contain the answer to the user's specific question, honestly state 'I don't know' or 'I don't have this information' (in the requested language) rather than inventing an answer.
- Do not invent hotel names, prices, availability, addresses, live booking status, or scores.
- Mention that the ratings are based on CMU TripAdvisor dataset evidence.
- Recommend the best hotel first.
- USE BEAUTIFUL MARKDOWN FORMATTING.
- Keep the answer highly professional, elegant, and concierge-level.

{style_instruction}
""".strip()


def generate_llm_answer(query, hotel_context, chat_history=None):
    if chat_history is None:
        chat_history = []
        
    lang_code = detect_language(query)
    language = language_name(lang_code)
    base_url = get_foundry_base_url()

    client = OpenAI(base_url=base_url, api_key="not-needed")
    model_id = get_available_model_id(client)

    prompt = build_prompt(query, language, hotel_context)

    if DEBUG:
        print("\nPhi cevap üretiyor...\n")

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{"role": "system", "content": prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": query}]),
            temperature=0.3,
            max_tokens=1000,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e: # pylint: disable=broad-exception-caught
        print("\n[HATA DETAYI] LLM Hatası:", str(e))
        return "Bağlantı hatası." if lang_code == "tr" else "Connection error."



def generate_followup_answer(query, last_context, lang_code, chat_history=None):
    if chat_history is None:
        chat_history = []
        
    language = language_name(lang_code)
    base_url = get_foundry_base_url()

    client = OpenAI(base_url=base_url, api_key="not-needed")
    model_id = get_available_model_id(client)

    system_prompt = f"""
You are TravelMind, a flawlessly professional, highly elite concierge-level, polite, and sophisticated local hotel recommendation assistant.
The user is asking a follow-up question (e.g. asking for another hotel option) about the previous results.

You must follow these rules:
- Read the retrieved hotel evidence provided below.
- If the user asks for another option, DO NOT talk about the very first recommended hotel again, focus on the OTHER hotel options in the evidence.
- Answer ONLY in {language}.
- Use ONLY the provided hotel evidence. Do not invent facts.
- {get_style_instruction(language)}
""".strip()

    user_prompt = f"""
User's query: {query}
Language: {language}

Retrieved hotel evidence (from previous search):
{last_context}
""".strip()

    if DEBUG:
        print("\nPhi alternatif üretiyor...\n")

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{"role": "system", "content": system_prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": user_prompt}]),
            temperature=0.3,
            max_tokens=1000,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e: # pylint: disable=broad-exception-caught
        print("\n[HATA DETAYI] LLM Hatası:", str(e))
        return "Bağlantı hatası." if lang_code == "tr" else "Connection error."


def generate_preference_refinement_answer(query, last_context, lang_code, chat_history=None):
    if chat_history is None:
        chat_history = []
        
    language = language_name(lang_code)
    base_url = get_foundry_base_url()

    client = OpenAI(base_url=base_url, api_key="not-needed")
    model_id = get_available_model_id(client)

    system_prompt = f"""
You are TravelMind, a flawlessly professional, highly elite concierge-level, polite, and sophisticated local hotel recommendation assistant.
The user is refining their preferences (e.g., cleanliness, location, service, room types) for their previous search.

You must follow these rules:
- Re-evaluate the provided hotel evidence based on the user's new priority.
- Highlight the hotel that best matches this new preference.
- Answer ONLY in {language}.
- Use ONLY the provided hotel evidence. Do not invent facts, prices, or live availability.
- {get_style_instruction(language)}
""".strip()

    user_prompt = f"""
User's new preference query: {query}
Language: {language}

Retrieved hotel evidence (from previous search):
{last_context}
""".strip()

    if DEBUG:
        print("\nPhi preference refinement cevap üretiyor...\n")

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{"role": "system", "content": system_prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": user_prompt}]),
            temperature=0.3,
            max_tokens=1000,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e: # pylint: disable=broad-exception-caught
        print("\n[HATA DETAYI] LLM Hatası:", str(e))
        return "Bağlantı hatası." if lang_code == "tr" else "Connection error."


def generate_score_explanation_answer(query, last_context, lang_code, chat_history=None):
    if chat_history is None:
        chat_history = []
        
    language = language_name(lang_code)
    base_url = get_foundry_base_url()

    client = OpenAI(base_url=base_url, api_key="not-needed")
    model_id = get_available_model_id(client)

    system_prompt = f"""
You are TravelMind, a flawlessly professional, highly elite concierge-level, polite, and sophisticated local hotel recommendation assistant.
The user is asking how the TravelMind suitability score was calculated for the previous hotels.

You must follow these rules:
- Read the Score Components and Score Weights from the provided hotel evidence (Hotel Card).
- Explain that the score is a weighted sum of these components, calculated by Python code (not by you).
- Do not hallucinate a different formula.
- Answer ONLY in {language}.
- Use elegant markdown formatting.
""".strip()

    user_prompt = f"""
User query: {query}
Language: {language}

Retrieved hotel evidence (contains Score Components and Weights):
{last_context}
""".strip()

    if DEBUG:
        print("\nPhi score explanation cevap üretiyor...\n")

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{"role": "system", "content": system_prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": user_prompt}]),
            temperature=0.3,
            max_tokens=1000,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e: # pylint: disable=broad-exception-caught
        print("\n[HATA DETAYI] LLM Hatası:", str(e))
        return "Bağlantı hatası." if lang_code == "tr" else "Connection error."


def generate_conversational_answer(query, lang_code, chat_history=None):
    if chat_history is None:
        chat_history = []
    language = language_name(lang_code)
    base_url = get_foundry_base_url()

    client = OpenAI(base_url=base_url, api_key="not-needed")

    model_id = get_available_model_id(client)

    system_prompt = f"""
You are TravelMind, a flawlessly professional, highly elite concierge-level, polite, and sophisticated AI assistant specialized in hotel and accommodation recommendations.

Guidelines:
1. Act naturally like a human assistant (e.g., ChatGPT). Respond intelligently and specifically to whatever the user says.
2. You only have data for hotel recommendations. You cannot help with flights, visas, restaurants, or itineraries.
3. If the user asks for a hotel without specifying a location, politely ask them which city or region they want to stay in.
4. NEVER invent or recommend specific hotel names. You do NOT have access to the database in this conversational mode. If you need to recommend hotels, you MUST ask the user for their preferred city so you can trigger the database search.
5. Always write your response in {language}.
6. Keep your answers concise, warm, and highly conversational.
""".strip()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{"role": "system", "content": system_prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": query}]),
            temperature=0.7,
            max_tokens=1000,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception: # pylint: disable=broad-exception-caught
        if lang_code == "tr":
            return "Su an baglanti kuramiyorum. Lutfen tekrar deneyin."
        return "I cannot connect right now. Please try again."

def get_llm_intent_and_location(query: str, chat_history: list) -> dict:
    base_url = get_foundry_base_url()
    client = OpenAI(base_url=base_url, api_key="not-needed")
    model_id = get_available_model_id(client)

    system_prompt = """
You are the Intent and Location Routing Engine for TravelMind AI.
Analyze the user's query and output a raw JSON object. Do not wrap it in markdown backticks.

Supported US Cities:
New York City, NY; Chicago, IL; San Francisco, CA; Boston, MA; Washington DC, DC; San Diego, CA; Dallas, TX; Houston, TX; Denver, CO; Los Angeles, CA; Seattle, WA; San Antonio, TX; Phoenix, AZ; Philadelphia, PA; Memphis, TN; Baltimore, MD; San Jose, CA; Detroit, MI; Austin, TX; Indianapolis, IN; Jacksonville, FL; Charlotte, NC; Columbus, OH; Fort Worth, TX; El Paso, TX.

Intent Categories:
- "hotel_search": User wants to search for a hotel in a specific city.
- "preference_refinement": User is refining their previous hotel search without mentioning a new city (e.g. "I want a cleaner one").
- "follow_up": User asks for another option from the previous search (e.g. "Başka seçenek var mı?").
- "score_explanation": User asks how the TravelMind score is calculated.
- "general_chat": User is greeting, chatting, or asking for help.
- "missing_location": User is asking for a hotel but hasn't mentioned ANY city/region.
- "unsupported_location": User is asking for a hotel in a city NOT in the Supported US Cities list (e.g. Paris, Miami, Istanbul).
- "out_of_scope": User asks about flights, restaurants, visas, etc.
- "exit": User wants to exit.

Output JSON format exactly:
{
  "intent": "<category>",
  "location": "<Formal name of the Supported City if detected, else null>"
}

Example 1:
Query: "Windy city'de temiz bir yer arıyorum"
Output: {"intent": "hotel_search", "location": "Chicago, IL"}

Example 2:
Query: "Paris'te lüks oteller"
Output: {"intent": "unsupported_location", "location": null}

Example 3:
Query: "Merhaba nasilsin?"
Output: {"intent": "general_chat", "location": null}
"""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{"role": "system", "content": system_prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": query}]),
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        return json.loads(content)
    except Exception as e:
        print(f"Error in intent routing: {e}")
        return {"intent": "general_chat", "location": None}


def main():
    console = Console()
    console.print("[bold cyan]TravelMind RAG - CMU + Phi Final Answer[/bold cyan]")
    console.print("[dim]" + ("-" * 60) + "[/dim]")

    current_lang = "tr"
    chat_history = []
    
    # State tracking
    last_results = None
    last_context = ""
    
    print("\n" + initial_welcome_message(current_lang))

    while True:
        query = input(input_prompt(current_lang)).strip()

        if not query:
            print(empty_query_message(current_lang))
            continue

        lang_code = detect_language(query, fallback_lang=current_lang)
        current_lang = lang_code

        if query.lower() in ["exit", "quit", "çık", "çıkış", "cıkıs", "cikis", "q"]:
            print(goodbye_message(lang_code))
            break

        router_res = get_llm_intent_and_location(query, chat_history)
        intent = router_res.get("intent", "general_chat")
        location = router_res.get("location", None)
        
        if DEBUG:
            print(f"LLM Router -> Intent: {intent}, Location: {location}")

        if intent == "exit":
            print(goodbye_message(lang_code))
            break
        if intent == "unsupported_location":
            answer = unsupported_location_message(lang_code)
            console.print(f"\n[bold green]TravelMind:[/bold green] {answer}")
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            chat_history = chat_history[-6:]
            continue
        if intent == "missing_location":
            if lang_code == "tr":
                print("\nTravelMind düşünüyor...")
            else:
                print("\nTravelMind is thinking...\n")
            answer = generate_conversational_answer(query, lang_code, chat_history)
            console.print(f"\n[bold green]TravelMind:[/bold green] {answer}")
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            chat_history = chat_history[-6:]
            continue
        if intent == "follow_up":
            if not last_results:
                if lang_code == "tr":
                    answer = "Lütfen önce bana bir otel araması yaptırın."
                else:
                    answer = "Please make a hotel search first."
                console.print(f"\n[bold green]TravelMind:[/bold green] {answer}")
                continue
            
            if lang_code == "tr":
                print("\nTravelMind alternatifleri değerlendiriyor...")
            else:
                print("\nTravelMind is evaluating alternatives...\n")
                
            answer = generate_followup_answer(query, last_context, lang_code, chat_history)
            console.print("\n[bold cyan]TravelMind Final Cevap[/bold cyan]" if lang_code == "tr" else "\n[bold cyan]TravelMind Final Answer[/bold cyan]")
            console.print("[dim]" + ("-" * 80) + "[/dim]")
            console.print(Markdown(answer))
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            chat_history = chat_history[-6:]
            continue
        if intent == "preference_refinement":
            if not last_results:
                if lang_code == "tr":
                    answer = "Lütfen önce bana bir otel araması yaptırın."
                else:
                    answer = "Please make a hotel search first."
                console.print(f"\n[bold green]TravelMind:[/bold green] {answer}")
                continue
            
            if lang_code == "tr":
                print("\nTravelMind tercihlerinize göre otelleri yeniden değerlendiriyor...")
            else:
                print("\nTravelMind is re-evaluating hotels based on your preferences...\n")
                
            answer = generate_preference_refinement_answer(query, last_context, lang_code, chat_history)
            console.print("\n[bold cyan]TravelMind Final Cevap[/bold cyan]" if lang_code == "tr" else "\n[bold cyan]TravelMind Final Answer[/bold cyan]")
            console.print("[dim]" + ("-" * 80) + "[/dim]")
            console.print(Markdown(answer))
            
            chat_history.append({"role": "user", "content": query})
            chat_history = chat_history[-6:]
            continue
        if intent == "score_explanation":
            if not last_results:
                if lang_code == "tr":
                    answer = "Lütfen önce bana bir otel araması yaptırın."
                else:
                    answer = "Please make a hotel search first."
                console.print(f"\n[bold green]TravelMind:[/bold green] {answer}")
                continue
            
            if lang_code == "tr":
                print("\nTravelMind skor hesaplamasını açıklıyor...")
            else:
                print("\nTravelMind is explaining the score calculation...\n")
                
            answer = generate_score_explanation_answer(query, last_context, lang_code, chat_history)
            console.print("\n[bold cyan]TravelMind Final Cevap[/bold cyan]" if lang_code == "tr" else "\n[bold cyan]TravelMind Final Answer[/bold cyan]")
            console.print("[dim]" + ("-" * 80) + "[/dim]")
            console.print(Markdown(answer))
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            chat_history = chat_history[-6:]
            continue

        if intent in [
            "greeting",
            "help",
            "out_of_scope",
            "general_chat",
        ]:
            if lang_code == "tr":
                print("\nTravelMind düşünüyor...")
            else:
                print("\nTravelMind is thinking...\n")

            answer = generate_conversational_answer(query, lang_code, chat_history)

            console.print(f"\n[bold green]TravelMind:[/bold green] {answer}")
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            chat_history = chat_history[-6:]
            
            continue

        if DEBUG:
            print(
                "\nCMU vector DB üzerinden oteller getiriliyor...\n"
                if lang_code == "tr"
                else "\nFetching hotels from CMU vector DB...\n"
            )

        os.environ["TRAVELMIND_RETRIEVAL_DEVICE"] = "cpu"

        search_query = query
        user_messages = [m["content"] for m in chat_history if m["role"] == "user"]
        if user_messages:
            search_query = user_messages[-1] + " " + query

        results = search(search_query, location_filter=location, top_k_hotels=TOP_K_RETRIEVAL)

        if not results:
            print(no_result_message(lang_code))
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": no_result_message(lang_code)})
            chat_history = chat_history[-6:]
            continue

        results = sorted(
            results,
            key=lambda result: calculate_recommendation_score(result)["score"],
            reverse=True,
        )

        selected_results = results[:TOP_K_FOR_LLM]

        hotel_context_str = ""
        for i, result in enumerate(selected_results, start=1):
            hotel_context_str += build_hotel_context(result, i) + "\n\n"

        # Update state
        last_results = results
        last_context = hotel_context_str

        answer = generate_llm_answer(query, hotel_context_str, chat_history)

        console.print("\n[bold cyan]TravelMind Final Cevap[/bold cyan]" if lang_code == "tr" else "\n[bold cyan]TravelMind Final Answer[/bold cyan]")
        console.print("[dim]" + ("-" * 80) + "[/dim]")
        
        md = Markdown(answer)
        console.print(md)
        
        console.print("\n[dim]Not:[/dim]" if lang_code == "tr" else "\n[dim]Note:[/dim]")
        console.print("[dim]- Retrieval ve skorlar CMU TripAdvisor datasetinden gelen chunk'lara dayanır.[/dim]" if lang_code == "tr" else "[dim]- Retrieval and scores are based on CMU TripAdvisor dataset chunks.[/dim]")
        console.print("[dim]- Yerel LLM yalnızca bu kanıtları doğal cevaba dönüştürür.[/dim]" if lang_code == "tr" else "[dim]- The local LLM only translates evidence into a natural answer.[/dim]")
        console.print("[dim]- Fiyat, canlı müsaitlik ve güncel rezervasyon bilgisi üretilmez.[/dim]" if lang_code == "tr" else "[dim]- Price, live availability, and booking info are not generated.[/dim]")

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        chat_history = chat_history[-6:]


if __name__ == "__main__":
    main()
