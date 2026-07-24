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

from openai import OpenAI, BadRequestError, APIConnectionError, APIError  # type: ignore
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

    try:
        models = client.models.list()
        model_ids = [model.id for model in models.data]
    except (APIConnectionError, APIError) as e:
        if DEBUG:
            print(f"Modeller listelenirken bağlantı hatası: {e}")
        return "local-model"

    if DEBUG:
        print("Bulunan modeller:")
        for model_id in model_ids:
            print("-", model_id)

    if not model_ids:
        return "local-model"

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


def stream_and_strip_think(response, lang_code):
    is_thinking = False
    buffer = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
            
        buffer += delta
        
        if not is_thinking:
            if "<think>" in buffer:
                is_thinking = True
                parts = buffer.split("<think>")
                if parts[0]:
                    yield {"type": "answer", "content": parts[0]}
                buffer = parts[1]
            else:
                if "<" in buffer:
                    idx = buffer.find("<")
                    possible_tag = buffer[idx:]
                    if "<think>".startswith(possible_tag):
                        if idx > 0:
                            yield {"type": "answer", "content": buffer[:idx]}
                            buffer = buffer[idx:]
                        continue
                
                yield {"type": "answer", "content": buffer}
                buffer = ""
                
        if is_thinking:
            if "</think>" in buffer:
                is_thinking = False
                parts = buffer.split("</think>")
                if parts[0]:
                    yield {"type": "think", "content": parts[0]}
                buffer = parts[1]
            else:
                if "<" in buffer:
                    idx = buffer.rfind("<")
                    possible_tag = buffer[idx:]
                    if "</think>".startswith(possible_tag):
                        if idx > 0:
                            yield {"type": "think", "content": buffer[:idx]}
                            buffer = buffer[idx:]
                        continue
                
                yield {"type": "think", "content": buffer}
                buffer = ""

    if buffer:
        if is_thinking:
            yield {"type": "think", "content": buffer}
        else:
            yield {"type": "answer", "content": buffer}

def generate_llm_answer(query, context_str, chat_history, location, lang_code="tr"):
    try:
        from openai import OpenAI
        base_url = get_foundry_base_url()
        client = OpenAI(base_url=base_url, api_key='not-needed')
        model_id = get_available_model_id(client)
        
        meta = load_hotel_metadata()
        total_hotels_in_city = sum(1 for h in meta.values() if h.get('city', '').lower() == location.lower()) if location else 0

        lang_map = {"en": "English", "tr": "Turkish", "de": "German", "fr": "French", "it": "Italian", "zh": "Chinese"}
        target_lang = lang_map.get(lang_code, "Turkish")
        
        no_chinese_rule = "- Asla Çince karakterler kullanma." if target_lang != "Chinese" else ""

        system_prompt = f"""Sen TravelMind — elit düzeyde profesyonel bir yapay zeka seyahat asistanısın. Müşterilere üst düzey, kusursuz ve sofistike bir deneyim sunmalısın.

CRITICAL RULES:
- THE TARGET LANGUAGE FOR YOUR RESPONSE IS STRICTLY: {target_lang}. YOU MUST TRANSLATE EVERYTHING (INCLUDING HEADINGS, LABELS, AND TEXT) TO {target_lang}.
- SADECE aşağıdaki Otel Verileri kısmındaki bilgilere dayanarak yanıt ver. Verilerde olmayan bir şeyi asla uydurma. Eğer verilen bağlam soruyu cevaplamak için yetersizse veya mevcut değilse, uydurma (halüsinasyon görme). Nazikçe 'Şu anki bilgilerimle bu konuda size yardımcı olamıyorum' veya 'Bununla ilgili yeterli bilgiye sahip değilim' şeklinde cevap ver (Hedef dilde).
- Fiyat veya bütçelerden asla bahsetme; bu konu sistemin kapsamı dışındadır.
- Önerdiğin otellerin Harita Bağlantısını (Google Maps linkini) her otelin açıklamasının sonuna tıklanabilir Markdown formatında ekle (Örn: `[Haritada Gör](link)`).
- Asla "karmaşık yorumlar", "karışık yorumlar" veya "çelişkili yorumlar" deme. Bunun yerine "Yorumlar bu konuda farklılık göstermektedir" veya "Ziyaretçi deneyimleri bu açıdan çeşitlilik göstermektedir" ifadelerini kullan (Tabii ki {target_lang} dilinde).
- Asla kendi içinde çelişme. 
- Eğer müşteri '{location}' şehrinde kaç otel bildiğini sorarsa: veritabanımızda tam olarak {total_hotels_in_city} adet seçkin otel var de.
{no_chinese_rule}
- YANITINA DOĞRUDAN BAŞLA. ASLA "Okay, the user said..." gibi iç ses analizleri yazma! Sadece müşterine hitap et.
- ZORUNLU KURAL: Yanıtına KESİNLİKLE `<think>` yazarak BAŞLAMALISIN! Başka hiçbir kelime ile başlama. İç sesini ve planlamanı bu etiket içinde yap, bittikten sonra `</think>` etiketini kapat ve müşteriye asıl elit yanıtını yaz.

YANIT FORMATI (Bu şablona sadık kal ama başlıkları {target_lang} diline ÇEVİR):
- Her otel için: 🏨 [Hotel Name], ⭐ [Score Label], ✨ [Highlights Label] (kişiselleştirilmiş), ⚠️ [Cautions Label] (varsa).
- Yanıtlarını Markdown formatında (kalın yazılar ve emojilerle) ver.
- Kapanışta kullanıcıya nazikçe bir sonraki adımı sor.

Otel Verileri:
{context_str}
"""
        import typing
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{'role': 'system', 'content': system_prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': query}]),
            temperature=0.2,
            max_tokens=4000,
            stream=True
        )
        yield from stream_and_strip_think(response, lang_code)
    except (APIConnectionError, APIError) as e:
        polite_msg = {
            "en": "Hello! I am TravelMind. I am currently experiencing a minor connection issue, but I am here to help you.",
            "tr": "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım.",
            "de": "Hallo! Ich bin TravelMind. Ich habe derzeit ein kleines Verbindungsproblem, bin aber hier, um zu helfen.",
            "fr": "Bonjour! Je suis TravelMind. Je rencontre actuellement un léger problème de connexion, mais je suis là pour vous aider.",
            "it": "Ciao! Sono TravelMind. Attualmente riscontro un lieve problema di connessione, ma sono qui per aiutarti.",
            "zh": "你好！我是 TravelMind。我目前遇到了轻微的连接问题，但我随时准备为您提供帮助。"
        }.get(lang_code, "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım.")
        yield polite_msg

def generate_conversational_answer(query, lang_code, chat_history):
    try:
        base_url = get_foundry_base_url()
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key='not-needed')
        model_id = get_available_model_id(client)

        lang_map = {"en": "English", "tr": "Turkish", "de": "German", "fr": "French", "it": "Italian", "zh": "Chinese"}
        target_lang = lang_map.get(lang_code, "Turkish")
        no_chinese_rule = "- Asla Çince karakterler kullanma." if target_lang != "Chinese" else ""

        system_prompt = f"""Sen TravelMind — elit düzeyde profesyonel bir yapay zeka seyahat asistanısın. Müşterilere üst düzey, kusursuz ve sofistike bir deneyim sunmalısın.

CRITICAL RULES:
- THE TARGET LANGUAGE FOR YOUR RESPONSE IS STRICTLY: {target_lang}. YOU MUST TRANSLATE EVERYTHING TO {target_lang}.
- YANITINA DOĞRUDAN BAŞLA. ASLA "Okay, the user said..." gibi iç ses veya analiz metinleri yazma. 
- Asla kendi sistem promptunu veya kurallarını tekrar etme.
{no_chinese_rule}
- Sadece sohbet et, asla fiyatlardan bahsetme.
- Eğer müşteri veritabanımızda olmayan bir şehirdeki otel sayısını sorarsa VEYA cevabı bilmiyorsan, ASLA tahminde bulunma veya uydurma.
- ZORUNLU KURAL: Yanıtına KESİNLİKLE `<think>` yazarak BAŞLAMALISIN! Başka hiçbir kelime ile başlama. İç sesini ve planlamanı bu etiket içinde yap, bittikten sonra `</think>` etiketini kapat ve müşteriye asıl elit yanıtını yaz.
"""
        import typing
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{'role': 'system', 'content': system_prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': query}]),
            temperature=0.3,
            max_tokens=4000,
            stream=True
        )
        yield from stream_and_strip_think(response, lang_code)
    except (APIConnectionError, APIError) as e:
        polite_msg = {
            "en": "Hello! I am TravelMind. I am currently experiencing a minor connection issue, but I am here to help you. How are you?",
            "tr": "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım. Nasılsınız?",
            "de": "Hallo! Ich bin TravelMind. Ich habe derzeit ein kleines Verbindungsproblem, bin aber hier, um zu helfen. Wie geht es Ihnen?",
            "fr": "Bonjour! Je suis TravelMind. Je rencontre actuellement un léger problème de connexion, mais je suis là pour vous aider. Comment allez-vous?",
            "it": "Ciao! Sono TravelMind. Attualmente riscontro un lieve problema di connessione, ma sono qui per aiutarti. Come stai?",
            "zh": "你好！我是 TravelMind。我目前遇到了轻微的连接问题，但我随时准备为您提供帮助。你好吗？"
        }.get(lang_code, "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım. Nasılsınız?")
        yield polite_msg

def generate_followup_answer(query, context_str, lang_code, chat_history):
    try:
        base_url = get_foundry_base_url()
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key='not-needed')
        model_id = get_available_model_id(client)

        lang_map = {"en": "English", "tr": "Turkish", "de": "German", "fr": "French", "it": "Italian", "zh": "Chinese"}
        target_lang = lang_map.get(lang_code, "Turkish")
        no_chinese_rule = "- Asla Çince karakterler kullanma." if target_lang != "Chinese" else ""

        system_prompt = f"""Sen TravelMind — elit düzeyde profesyonel bir yapay zeka seyahat asistanısın. Müşterilere üst düzey, kusursuz ve sofistike bir deneyim sunmalısın.

CRITICAL RULES:
- THE TARGET LANGUAGE FOR YOUR RESPONSE IS STRICTLY: {target_lang}. YOU MUST TRANSLATE EVERYTHING (INCLUDING HEADINGS, LABELS, AND TEXT) TO {target_lang}.
- SADECE sana sağlanan bağlamdaki (context) verileri kullan. Eğer verilen bağlam soruyu cevaplamak için yetersizse veya mevcut değilse, uydurma (halüsinasyon görme). Nazikçe 'Şu anki bilgilerimle bu konuda size yardımcı olamıyorum' veya 'Bununla ilgili yeterli bilgiye sahip değilim' şeklinde cevap ver (Hedef dilde).
- Önerdiğin otellerin Harita Bağlantısını (Google Maps linkini) her otelin açıklamasının sonuna tıklanabilir Markdown formatında ekle (Örn: `[Haritada Gör](link)`).
- Asla "karmaşık yorumlar", "karışık yorumlar" veya "çelişkili yorumlar" deme. Bunun yerine "Yorumlar bu konuda farklılık göstermektedir" veya "Ziyaretçi deneyimleri bu açıdan çeşitlilik göstermektedir" ifadelerini kullan (Tabii ki {target_lang} dilinde).
- Asla kendi içinde çelişme. 
{no_chinese_rule}
- YANITINA DOĞRUDAN BAŞLA. ASLA "Okay, the user said..." gibi iç ses analizleri yazma!
- ZORUNLU KURAL: Yanıtına KESİNLİKLE `<think>` yazarak BAŞLAMALISIN! Başka hiçbir kelime ile başlama. İç sesini ve planlamanı bu etiket içinde yap, bittikten sonra `</think>` etiketini kapat ve müşteriye asıl elit yanıtını yaz.

YANIT FORMATI:
- Sıcak, prestijli bir concierge gibi doğrudan cevap ver.
- Eğer skor soruyorsa: robotik liste değil, doğal bir sohbet havasında açıkla.
- Yanıtının sonunda konuşmayı ilerletecek nazik bir soru sor.

Mevcut Otel Verileri:
{context_str}
"""
        import typing
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{'role': 'system', 'content': system_prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': query}]),
            temperature=0.4,
            max_tokens=4000,
            stream=True
        )
        yield from stream_and_strip_think(response, lang_code)
    except (APIConnectionError, APIError) as e:
        polite_msg = {
            "en": "Hello! I am TravelMind. I am currently experiencing a minor connection issue, but I am here to help you.",
            "tr": "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım.",
            "de": "Hallo! Ich bin TravelMind. Ich habe derzeit ein kleines Verbindungsproblem, bin aber hier, um zu helfen.",
            "fr": "Bonjour! Je suis TravelMind. Je rencontre actuellement un léger problème de connexion, mais je suis là pour vous aider.",
            "it": "Ciao! Sono TravelMind. Attualmente riscontro un lieve problema di connessione, ma sono qui per aiutarti.",
            "zh": "你好！我是 TravelMind。我目前遇到了轻微的连接问题，但我随时准备为您提供帮助。"
        }.get(lang_code, "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım.")
        yield polite_msg


def consume_generator(generator, console):
    import sys
    full_answer = ""
    console.print("\n[bold green]TravelMind:[/bold green] ", end="")
    for chunk in generator:
        if isinstance(chunk, dict):
            if chunk["type"] == "think":
                sys.stdout.write(f"\033[90m{chunk['content']}\033[0m")
                sys.stdout.flush()
            else:
                full_answer += chunk["content"]
                sys.stdout.write(chunk["content"])
                sys.stdout.flush()
        else:
            full_answer += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
    print()
    return full_answer

def get_llm_intent_and_location(query: str, chat_history: list) -> dict:
    try:
        base_url = get_foundry_base_url()
        from openai import OpenAI
        import typing, json
        client = OpenAI(base_url=base_url, api_key="not-needed")
        model_id = get_available_model_id(client)

        system_prompt = """
You are the Intent, Location, and Keyword Routing Engine for TravelMind AI.
Analyze the user's query and output a raw JSON object. Do not wrap it in markdown backticks.

Supported US Cities:
New York City, NY; Chicago, IL; San Francisco, CA; Boston, MA; Washington DC, DC; San Diego, CA; Dallas, TX; Houston, TX; Denver, CO; Los Angeles, CA; Seattle, WA; San Antonio, TX; Phoenix, AZ; Philadelphia, PA; Memphis, TN; Baltimore, MD; San Jose, CA; Detroit, MI; Austin, TX; Indianapolis, IN; Jacksonville, FL; Charlotte, NC; Columbus, OH; Fort Worth, TX; El Paso, TX.

Intent Categories:
- "hotel_search": User mentions a city and asks about hotels.
- "preference_refinement": User is refining their previous hotel search without mentioning a new city (e.g. "I want a cleaner one").
- "follow_up": User asks for another option from the previous search (e.g. "Başka seçenek var mı?").
- "score_explanation": User asks how the TravelMind score is calculated.
- "general_chat": User is greeting, chatting, or asking for help. Examples: "merhaba", "selam", "hello", "hi". If the user JUST says a greeting and nothing about a city, it MUST be general_chat.
- "missing_location": User is asking for a hotel but hasn't mentioned ANY city/region.
- "unsupported_location": User is asking for a hotel in a city NOT in the Supported US Cities list (e.g. Paris, Miami, Istanbul).
- "out_of_scope": User asks about flights, restaurants, visas, etc.
- "exit": User wants to exit.

Output JSON format exactly:
{
  "intent": "<category>",
  "location": "<Formal name of the Supported City if detected, else null>",
  "filters": {
    "pool": <boolean, true if user asks for pool/swimming>,
    "wifi": <boolean, true if user asks for wifi/internet>,
    "breakfast": <boolean, true if user asks for breakfast>,
    "pet": <boolean, true if user asks for pet-friendly>,
    "gym": <boolean, true if user asks for gym/fitness>,
    "parking": <boolean, true if user asks for parking>,
    "restaurant": <boolean, true if user asks for a restaurant/dining>,
    "bar": <boolean, true if user asks for a bar/lounge>,
    "spa": <boolean, true if user asks for a spa>,
    "room_service": <boolean, true if user asks for room service>,
    "business_center": <boolean, true if user asks for a business center/work space>,
    "tv": <boolean, true if user asks for a TV>,
    "smoke_free": <boolean, true if user asks for a smoke-free/non-smoking room>
  }
}

CRITICAL: YOU MUST OUTPUT ONLY RAW JSON. DO NOT WRITE ANY OTHER TEXT. NO EXPLANATIONS.

Example 1:
Query: "Windy city'de ucuz havuzlu bir yer arıyorum"
Output: {"intent": "hotel_search", "location": "Chicago, IL", "filters": {"pool": true, "wifi": false, "breakfast": false, "pet": false, "gym": false, "parking": false, "restaurant": false, "bar": false, "spa": false, "room_service": false, "business_center": false, "tv": false, "smoke_free": false}}

Example 2:
Query: "Paris'te lüks oteller"
Output: {"intent": "unsupported_location", "location": null, "filters": {"pool": false, "wifi": false, "breakfast": false, "pet": false, "gym": false, "parking": false, "restaurant": false, "bar": false, "spa": false, "room_service": false, "business_center": false, "tv": false, "smoke_free": false}}
"""
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{"role": "system", "content": system_prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": query}]),
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        import re
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
                
        if not parsed:
            parsed = {"intent": "general_chat", "location": None, "filters": {}}
            
        # Hard fallback heuristic
        q_lower = query.lower()
        import difflib
        is_hotel_query = False
        words = q_lower.replace("'", " ").replace('"', " ").split()
        if "otel" in q_lower or "hotel" in q_lower or "kalacak" in q_lower or "konaklama" in q_lower:
            is_hotel_query = True
        else:
            if difflib.get_close_matches("otel", words, n=1, cutoff=0.7) or difflib.get_close_matches("hotel", words, n=1, cutoff=0.7):
                is_hotel_query = True

        if parsed.get("intent") in ["general_chat", "missing_location"] and is_hotel_query:
            cities_map = {
                "dallas": "Dallas, TX", "chicago": "Chicago, IL", "new york": "New York City, NY",
                "san francisco": "San Francisco, CA", "boston": "Boston, MA", "washington": "Washington DC, DC",
                "san diego": "San Diego, CA", "houston": "Houston, TX", "denver": "Denver, CO",
                "los angeles": "Los Angeles, CA", "seattle": "Seattle, WA", "san antonio": "San Antonio, TX",
                "phoenix": "Phoenix, AZ", "philadelphia": "Philadelphia, PA", "memphis": "Memphis, TN",
                "baltimore": "Baltimore, MD", "san jose": "San Jose, CA", "detroit": "Detroit, MI",
                "austin": "Austin, TX", "indianapolis": "Indianapolis, IN", "jacksonville": "Jacksonville, FL",
                "charlotte": "Charlotte, NC", "columbus": "Columbus, OH", "fort worth": "Fort Worth, TX",
                "el paso": "El Paso, TX"
            }
            import difflib
            words = q_lower.split()
            bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
            all_tokens = words + bigrams
            
            best_match = None
            best_val = None
            
            for c_key, c_val in cities_map.items():
                if c_key in q_lower:
                    best_match = c_key
                    best_val = c_val
                    break
                
                matches = difflib.get_close_matches(c_key, all_tokens, n=1, cutoff=0.75)
                if matches:
                    best_match = matches[0]
                    best_val = c_val
                    break
                    
            if best_match:
                parsed["intent"] = "hotel_search"
                parsed["location"] = best_val
                    
        return parsed
    except Exception as e:
        print(f"Error in intent routing: {e}")
        return {"intent": "general_chat", "location": None, "filters": {}}

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

        try:
            results = search(search_query, location_filter=location, top_k_hotels=TOP_K_RETRIEVAL)
        except Exception as e:
            print(f"RAG Retrieval failed: {e}")
            results = []

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

        generator = generate_llm_answer(query, hotel_context_str, chat_history, location, lang_code)
        answer = consume_generator(generator, console)
        
        console.print("\n[dim]Not:[/dim]" if lang_code == "tr" else "\n[dim]Note:[/dim]")
        console.print("[dim]- Retrieval ve skorlar CMU TripAdvisor datasetinden gelen chunk'lara dayanır.[/dim]" if lang_code == "tr" else "[dim]- Retrieval and scores are based on CMU TripAdvisor dataset chunks.[/dim]")
        console.print("[dim]- Yerel LLM yalnızca bu kanıtları doğal cevaba dönüştürür.[/dim]" if lang_code == "tr" else "[dim]- The local LLM only translates evidence into a natural answer.[/dim]")
        console.print("[dim]- Fiyat, canlı müsaitlik ve güncel rezervasyon bilgisi üretilmez.[/dim]" if lang_code == "tr" else "[dim]- Price, live availability, and booking info are not generated.[/dim]")

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        chat_history = chat_history[-6:]


if __name__ == "__main__":
    main()
