import json
import typing
from openai import OpenAI, APIConnectionError, APIError
import os
import sys
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

from openai import APIConnectionError, APIError

from cmu_retrieve import search
from travelmind_scoring import (
    calculate_travelmind_score,
    build_strengths,
    build_cautions,
)
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
        if "qwen3-4b" in model_id.lower() or "qwen3.5-2b" in model_id.lower() or "qwen" in model_id.lower():
            if DEBUG:
                print("Kullanılacak model:", model_id)
            return model_id

    selected_model = model_ids[0]
    if DEBUG:
        print("Qwen modeli bulunamadı, ilk model kullanılacak:", selected_model)
    return selected_model

def verbalize_amenity(feature_name, status, language="en"):
    is_tr = str(language).lower().startswith("tr")
    fname_tr = {"breakfast": "kahvaltı", "pool": "havuz", "wifi": "Wi-Fi", "wheelchair_accessible": "tekerlekli sandalye erişimi", "parking": "otopark"}.get(feature_name, feature_name)
    fname_en = feature_name.replace("_", " ").lower()
    
    if status == "YES":
        return f"Bu otelde {fname_tr} bulunuyor." if is_tr else f"This hotel offers {fname_en}."
    elif status == "NO":
        return f"Otelde {fname_tr} hizmeti maalesef yok." if is_tr else f"Unfortunately, this hotel does not have {fname_en}."
    else:
        return ""

def verbalize_room_info(room_type, status, language="en"):
    is_tr = str(language).lower().startswith("tr")
    rname_tr = {"single_room": "tek kişilik oda", "double_room": "çift kişilik oda", "suite": "süit oda"}.get(room_type, room_type)
    rname_en = room_type.replace("_", " ").lower()
    
    if status == "YES":
        return f"Ayrıca {rname_tr} seçenekleri mevcut." if is_tr else f"Also, {rname_en} options are available."
    elif status == "NO":
        return f"Şu anki bilgilere göre {rname_tr} seçeneği görünmüyor." if is_tr else f"Based on current information, {rname_en} options are not visible."
    else:
        return ""



def build_hotel_context(query, card, index):
    hotel_name = card.get("hotel_name", "")
    location = card.get("location", "")
    hotel_class = card.get("hotel_class", "")
    total_review_count = card.get("review_count", "Unknown")
    
    scoring = {
        "travelmind_score": card.get("travelmind_score", 0),
        "rank_score": card.get("rank_score", 0)
    }

    strengths = card.get("strengths", [])
    cautions = card.get("cautions", [])

    room_info_dict = card.get("room_info", {})
    has_single_room = room_info_dict.get("single_room") == "YES"
    has_double_room = room_info_dict.get("double_room") == "YES"
    has_suite = room_info_dict.get("suite") == "YES"
    booking_rooms = room_info_dict.get("booking_room_types", [])
    room_types_meta = room_info_dict.get("room_types", [])
    
    room_info_list = []
    if has_single_room:
        room_info_list.append("single rooms")
    if has_double_room:
        room_info_list.append("double rooms")
    if has_suite:
        room_info_list.append("suites (suit oda)")
        
    if room_types_meta:
        for rt in room_types_meta:
            if rt.lower() not in " ".join(room_info_list).lower():
                room_info_list.append(str(rt))
                
    if booking_rooms:
        for br in booking_rooms:
            if br.lower() not in " ".join(room_info_list).lower():
                room_info_list.append(str(br))
        
    room_info = "Unknown / Not explicitly guaranteed in this dataset chunk."
    if room_info_list:
        room_info = "Available Rooms: " + ", ".join(room_info_list)
        
    amenities = card.get("amenities", {})
    confirmed_amenities = []
    for k, v in amenities.items():
        if v == "YES":
            confirmed_amenities.append(k)
        elif k == "other" and isinstance(v, list):
            confirmed_amenities.extend(v)
            
    amenities_str = "None specifically confirmed"
    if confirmed_amenities:
        amenities_str = ", ".join(confirmed_amenities)

    exact_address = "Not Available"
    if card.get("map_link") and card.get("map_link") != "UNKNOWN":
        exact_address = "Available via map link"


    context = f"""
Hotel Card {index}:
- Hotel name: {hotel_name}
- Location: {location}
- Hotel Class (Star Rating): {hotel_class}
- Phone Number: {card.get('phone', 'Unknown')}
- Exact Address: {exact_address}
- TravelMind suitability score: {scoring.get("travelmind_score", 0):.1f}/100
- Total review count: {total_review_count}
- Strengths: {', '.join(strengths) if strengths else 'None specific'}
- Cautions: {', '.join(cautions) if cautions else 'None specific'}
- Room Types Available: {room_info}
- Amenities: {amenities_str}
- Score Components: {str(scoring.get("components", []))}
- Review Excerpt: {card.get('chunk_text', '')[:1200]}
"""

    return context.strip()





def stream_and_strip_think(response, lang_code):
    is_thinking = False
    buffer = ""
    answer_buffer = ""
    
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
                    ans = parts[0].replace("`", "")
                    if ans:
                        answer_buffer += ans
                        if answer_buffer.strip():
                            yield {"type": "answer", "content": answer_buffer}
                        answer_buffer = ""
                buffer = parts[1]
            else:
                if "<" in buffer:
                    idx = buffer.find("<")
                    possible_tag = buffer[idx:]
                    if "<think>".startswith(possible_tag):
                        if idx > 0:
                            ans = buffer[:idx].replace("`", "")
                            answer_buffer += ans
                            buffer = buffer[idx:]
                        continue
                
                ans = buffer.replace("`", "")
                answer_buffer += ans
                buffer = ""
                
                if any(punct in answer_buffer for punct in [". ", "! ", "? ", "\n"]):
                    last_idx = max(answer_buffer.rfind(". "), answer_buffer.rfind("! "), answer_buffer.rfind("? "), answer_buffer.rfind("\n"))
                    if last_idx != -1:
                        split_point = last_idx + 1 if answer_buffer[last_idx] == "\n" else last_idx + 2
                        complete_part = answer_buffer[:split_point]
                        answer_buffer = answer_buffer[split_point:]
                        
                        if complete_part.strip():
                            yield {"type": "answer", "content": complete_part}
                        else:
                            yield {"type": "answer", "content": complete_part}
                
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
            answer_buffer += buffer.replace("`", "")
            
    if answer_buffer.strip():
        yield {"type": "answer", "content": answer_buffer}
    elif answer_buffer:
        yield {"type": "answer", "content": answer_buffer}

def generate_llm_answer(query, hotel_context_str, chat_history, location, lang_code="tr", hotel_cards=None):
    try:
                        
        base_url = get_foundry_base_url()
        client = OpenAI(base_url=base_url, api_key='not-needed', timeout=25.0)
        model_id = get_available_model_id(client)
            
        meta = load_hotel_metadata()
        total_hotels_in_city = sum(1 for h in meta.values() if h.get('city', '').lower() == location.lower()) if location else 0

        lang_map = {"en": "English", "tr": "Turkish"}
        target_lang = lang_map.get(lang_code, "Turkish")
        
    

        import prompt_builders
        style_instruction = prompt_builders.get_style_instruction(target_lang)
        prompt = prompt_builders.build_final_answer_prompt(
            target_language=target_lang,
            intent="hotel_search",
            requested_location=location,
            hotel_context_str=hotel_context_str,
            total_hotels_in_city=total_hotels_in_city,
            style_instruction=style_instruction
        )
        
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{'role': 'system', 'content': prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': query}]),
            temperature=0.0,
            max_tokens=4000,
            stream=True
        )
        yield from stream_and_strip_think(response, lang_code)
    except Exception as e:
        yield {"type": "answer", "content": safe_card_based_fallback_answer(hotel_cards=hotel_cards if hotel_cards else [], language=lang_code)}


def generate_conversational_answer(query, lang_code, chat_history):
    try:
        base_url = get_foundry_base_url()
        client = OpenAI(base_url=base_url, api_key='not-needed', timeout=25.0)
        model_id = get_available_model_id(client)

        lang_map = {"en": "English", "tr": "Turkish", "de": "German", "fr": "French", "it": "Italian", "zh": "Chinese"}
        target_lang = lang_map.get(lang_code, "Turkish")
        import prompt_builders
        prompt = prompt_builders.build_conversational_answer_prompt(
            target_language=target_lang
        )

        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{'role': 'system', 'content': prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': query}]),
            temperature=0.3,
            max_tokens=4000,
            stream=True
        )
        yield from stream_and_strip_think(response, lang_code)
    except Exception as e:
        polite_msg = {
            "en": "Hello! I am TravelMind. I am currently experiencing a minor connection issue, but I am here to help you. How are you?",
            "tr": "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım. Nasılsınız?"
        }.get(lang_code, "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım. Nasılsınız?")
        yield polite_msg

def generate_followup_answer(query, context_str, lang_code, chat_history):
    try:
        from hotel_card_builder import build_hotel_cards
        
        base_url = get_foundry_base_url()
        client = OpenAI(base_url=base_url, api_key='not-needed')
        model_id = get_available_model_id(client)

        lang_map = {"en": "English", "tr": "Turkish"}
        target_lang = lang_map.get(lang_code, "Turkish")
        
        import prompt_builders
        style_instruction = prompt_builders.get_style_instruction(target_lang)
        prompt = prompt_builders.build_followup_prompt(
            target_language=target_lang,
            hotel_context_str=context_str if isinstance(context_str, str) else "",
            style_instruction=style_instruction
        )

        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{'role': 'system', 'content': prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': query}]),
            temperature=0.0,
            max_tokens=4000,
            stream=True
        )
        yield from stream_and_strip_think(response, lang_code)
    except Exception as e:
        polite_msg = {
            "en": "Hello! I am TravelMind. I am currently experiencing a minor connection issue, but I am here to help you.",
            "tr": "Merhaba! Ben TravelMind. Şu an sunucularımla küçük bir bağlantı sorunu yaşıyorum ama size yardım etmek için buradayım."
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

def fast_route_query(user_query, session_state=None) -> dict:
    import time
    start_time = time.time()
    
    q_lower = user_query.lower().strip()
    result = None
    
    # 0. Check if user is referring to a specific hotel from the last search
    if session_state and "last_hotel_cards" in session_state:
        for card in session_state.get("last_hotel_cards", []):
            hname = str(card.get("hotel_name", "")).lower().strip()
            # Remove common prefixes like 'the ' to make matching more robust
            if hname.startswith("the "):
                hname = hname[4:].strip()
            if len(hname) > 3 and hname in q_lower:
                print(f"[ROUTER] Caught context hotel name: {card.get('hotel_name')}")
                return {"intent": "specific_hotel_info", "requested_hotel_name": card.get("hotel_name")}
    
    # Check if a supported city is mentioned
    supported_cities = ["dallas", "chicago", "new york", "san francisco", "boston", "washington", "san diego", "houston", "denver", "los angeles", "seattle", "san antonio", "phoenix", "philadelphia", "memphis", "baltimore", "san jose", "detroit", "austin", "indianapolis", "jacksonville", "charlotte", "columbus", "fort worth", "el paso"]
    has_city = any(c in q_lower for c in supported_cities)
    
    # 1. Price check
    if any(p in q_lower for p in ["price", "how much", "per night", "fiyat", "gecelik", "ne kadar"]):
        result = {"intent": "price_question"}
        
    # 2. Pool follow-up
    elif not has_city and any(p in q_lower for p in ["pool", "havuz"]):
        result = {"intent": "followup_pool"}
        
    # 3. Breakfast follow-up
    elif not has_city and any(b in q_lower for b in ["breakfast", "kahvaltı"]):
        result = {"intent": "followup_breakfast"}
        
    # 4. Other hotel follow-up
    elif not has_city and any(o in q_lower for o in ["other hotel", "another hotel", "başka", "diğer", "sıradaki"]):
        result = {"intent": "followup_other_hotel"}
        
    # 5. Score / Class explanation
    elif any(s in q_lower for s in ["score", "skor", "puan", "hesapla", "calculate", "hesaplanır", "class", "sınıf"]):
        if any(s in q_lower for s in ["how", "nasıl", "what", "nedir"]):
            if "class" in q_lower or "sınıf" in q_lower:
                result = {"intent": "class_explanation"}
            else:
                result = {"intent": "score_explanation"}

    # 6. Unsupported Locations
    unsupported_cities = ["paris", "istanbul", "vienna", "miami", "las vegas", "london", "tokyo", "rome", "berlin", "madrid", "londra", "wien", "viyana", "roma"]
    if result is None:
        for uc in unsupported_cities:
            if uc in q_lower:
                result = {"intent": "unsupported_location", "location": uc.capitalize()}
                break
                
    # 7. Supported city + hotel/amenity signal -> hotel_search
    if result is None and has_city:
        hotel_signals = ["hotel", "otel", "konaklama", "stay", "room", "kahvalt", "breakfast", "wifi", "wi-fi", "pool", "havuz", "central", "merkezi", "clean", "temiz", "öner", "suggest", "recommend", "suite", "suit", "single", "double", "tek kişilik", "çift kişilik"]
        specific_signals = ["hakkında", "bilgi ver", "tell me about", "about", "oteli", "otelini"]
        
        found_city = next((c for c in supported_cities if c in q_lower), None)
        
        reqs = {}
        if "kahvalt" in q_lower or "breakfast" in q_lower:
            reqs["breakfast"] = "REQUIRED"
        if "wifi" in q_lower or "wi-fi" in q_lower:
            reqs["wifi"] = "REQUIRED"
        if "pool" in q_lower or "havuz" in q_lower:
            reqs["pool"] = "REQUIRED"
        if "suite" in q_lower or "suit" in q_lower:
            reqs["suite"] = "REQUIRED"
        if "single" in q_lower or "tek kişi" in q_lower:
            reqs["single_room"] = "REQUIRED"
        if "double" in q_lower or "çift kişi" in q_lower:
            reqs["double_room"] = "REQUIRED"
            
        if any(s in q_lower for s in specific_signals):
            result = {"intent": "specific_hotel_info", "city": found_city.capitalize(), "requirements": reqs}
        elif any(s in q_lower for s in hotel_signals):
            result = {"intent": "hotel_search", "city": found_city.capitalize(), "requirements": reqs}
                
    if result:
        print(f"[TIMING] fast_route_query matched '{result['intent']}' in {time.time() - start_time:.3f}s")
        return result
        
    print(f"[TIMING] fast_route_query fell through in {time.time() - start_time:.3f}s")
    return None

def get_llm_intent_and_location(query: str, chat_history: list) -> dict:
    q_lower = query.lower().strip()
    
    # Fast-path for greetings
    greetings = ["merhaba", "selam", "hello", "hi", "nasılsın", "naber", "iyi günler", "kolay gelsin", "hey", "merhaba nasılsın"]
    if q_lower in greetings or any(q_lower == g for g in greetings) or (len(q_lower.split()) <= 2 and any(g in q_lower for g in greetings)):
        return {"intent": "general_chat", "location": None, "filters": {}}

    try:
        base_url = get_foundry_base_url()
        import typing, json
        client = OpenAI(base_url=base_url, api_key="not-needed")
        model_id = get_available_model_id(client)

        import prompt_builders
        system_prompt = prompt_builders.build_router_system_prompt()
        response = client.chat.completions.create(
            model=model_id,
            messages=typing.cast(typing.Any, [{"role": "system", "content": system_prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": query}]),
            temperature=0.0,
            stream=True
        )
        content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        match = re.search(r'\{.*\}', content, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
                
        if not parsed:
            parsed = {"intent": "general_chat", "location": None, "query_requirements": {}}
            
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
            else:
                parsed["intent"] = "missing_location"
                parsed["location"] = None
        if parsed.get("intent") == "hotel_search":
            if not parsed.get("location") or str(parsed.get("location")).strip() == "":
                parsed["intent"] = "missing_location"
            else:
                loc_lower = str(parsed["location"]).lower()
                cities_map_keys = [
                    "dallas", "chicago", "new york", "san francisco", "boston", "washington", 
                    "san diego", "houston", "denver", "los angeles", "seattle", "san antonio", 
                    "phoenix", "philadelphia", "memphis", "baltimore", "san jose", "detroit", 
                    "austin", "indianapolis", "jacksonville", "charlotte", "columbus", 
                    "fort worth", "el paso"
                ]
                is_supported = any(c in loc_lower for c in cities_map_keys)
                if not is_supported:
                    parsed["intent"] = "unsupported_location"
                
        return parsed
    except Exception as e:
        print(f"Error in intent routing: {e}")
        return {"intent": "general_chat", "location": None, "query_requirements": {}}

def generate_out_of_scope_answer(user_query=None, language="en", chat_history=None):
    if language and str(language).lower().startswith("tr"):
        return (
            "TravelMind şu anda yalnızca desteklenen şehirlerdeki otel ve konaklama önerileri için çalışır. "
            "Fiyat, canlı müsaitlik, rezervasyon, uçuş, vize veya genel seyahat planı bilgisi sağlamaz."
        )
    return (
        "TravelMind currently supports hotel and accommodation recommendations only for supported cities. "
        "It does not provide live prices, availability, booking, flight, visa, or general travel planning information."
    )

def safe_card_based_fallback_answer(
    user_query=None,
    hotel_cards=None,
    query_requirements=None,
    city=None,
    language="en"
):
    is_tr = str(language).lower().startswith("tr")
    
    if not hotel_cards:
        if is_tr:
            return "Mevcut TravelMind verilerinde uygun otel seçeneği bulamadım."
        else:
            return "Based on the current TravelMind data, I could not find matching hotel options."
            
    best_card = hotel_cards[0]
    hotel_name = best_card.get("hotel_name", "UNKNOWN")
    score = best_card.get("travelmind_score", best_card.get("rank_score", 0.0))
    
    breakfast_status = best_card.get("amenities", {}).get("breakfast", "UNKNOWN")
    bf_sent = verbalize_amenity("breakfast", breakfast_status, language)
        
    sr_status = best_card.get("room_info", {}).get("single_room", "UNKNOWN")
    sr_sent = verbalize_room_info("single_room", sr_status, language)

    if city is None and best_card.get("location"):
        city = best_card.get("location").split(",")[0]

    if is_tr:
        city_str_tr = f"{city} bölgesindeki" if city else "bu bölgedeki"
        base_msg = f"Sizin için {city_str_tr} en uygun otelleri inceledim. Gözüme ilk çarpan seçenek {hotel_name} oldu. Bu otelin TravelMind uygunluk skoru 100 üzerinden {score:.1f}."
        if bf_sent:
            base_msg += f" {bf_sent}"
        if sr_sent:
            base_msg += f" {sr_sent}"
        return base_msg
    else:
        city_str_en = f" in {city}" if city else ""
        base_msg = f"I found some great hotel options{city_str_en} for you. The strongest match is {hotel_name}, with a TravelMind score of {score:.1f}/100."
        if bf_sent:
            base_msg += f" {bf_sent}"
        if sr_sent:
            base_msg += f" {sr_sent}"
        return base_msg

