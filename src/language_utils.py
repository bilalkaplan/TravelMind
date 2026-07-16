import re
def detect_language(query: str, fallback_lang: str = "en") -> str:
    text = query.strip()
    lowered = text.lower()

    english_signals = [
        "hello",
        "hi",
        "i want",
        "i need",
        "looking for",
        "hotel in",
        "clean hotel",
        "good reviews",
        "what can you do",
        "help",
    ]

    turkish_chars = "çğıöşüÇĞİÖŞÜ"

    turkish_words = {
        "merhaba",
        "selam",
        "otel",
        "oteli",
        "arıyorum",
        "ariyorum",
        "istiyorum",
        "temiz",
        "merkezi",
        "konum",
        "konumu",
        "yorum",
        "yorumları",
        "iyi",
        "kötü",
        "şehir",
        "bölge",
    }

    for signal in english_signals:
        if re.search(r"\b" + re.escape(signal) + r"\b", lowered):
            return "en"

    if any(char in text for char in turkish_chars):
        return "tr"

    words = set(lowered.replace("'", " ").replace("’", " ").split())
    if words.intersection(turkish_words):
        return "tr"

    return fallback_lang


def language_name(lang_code: str) -> str:
    if lang_code == "tr":
        return "Turkish"
    return "English"




def greeting_message(lang_code: str) -> str:
    if lang_code == "tr":
        return "Merhaba, ben TravelMind. Otel ve konaklama seçimi konusunda yardımcı olabilirim. Bana şehir veya ülke bilgisiyle birlikte temizlik, merkezi konum, yorum kalitesi, servis ya da oda konforu gibi tercihlerini yazabilirsin. Örneğin: 'Chicago'da temiz, merkezi ve iyi yorumları olan bir otel arıyorum.'"
    return "Hello, I’m TravelMind. I can help you choose hotels based on location, cleanliness, reviews, service, and room comfort. Please include a city, country, or region in your request. For example: 'I want a clean, central hotel in Chicago with good reviews.'"


def help_message(lang_code: str) -> str:
    return greeting_message(lang_code)


def missing_location_message(lang_code: str) -> str:
    if lang_code == "tr":
        return "Otel önerisi yapabilmem için şehir bilgisi gerekiyor. Şu an yalnızca ABD'deki bazı büyük şehirleri destekliyorum. Örneğin 'Chicago'da temiz ve merkezi bir otel arıyorum' gibi bir sorgu yazabilirsin."
    return "I need a city to recommend hotels. I currently only support selected major US cities. For example: 'I want a clean and central hotel in Chicago.'"


def unsupported_location_message(lang_code: str) -> str:
    if lang_code == "tr":
        return "Bu konum mevcut CMU TripAdvisor veri setimde bulunmuyor. Şu anda yalnızca ABD'deki belirli şehirler için otel önerisi yapabiliyorum. Desteklenen bazı şehirler: New York City, Chicago, San Francisco, Boston, Washington DC, Los Angeles, Seattle, Dallas ve Philadelphia."
    return "This location is not available in my current CMU TripAdvisor dataset. I can currently recommend hotels only for selected U.S. cities, such as New York City, Chicago, San Francisco, Boston, Washington DC, Los Angeles, Seattle, Dallas, and Philadelphia."

def out_of_scope_message(lang_code: str) -> str:
    if lang_code == "tr":
        return "TravelMind şu anda yalnızca otel ve konaklama seçimi için tasarlandı. Uçak bileti, vize, restoran veya gezi rotası önerisi üretmiyorum. Ancak belirli bir şehir için otel tercihlerini yazarsan yardımcı olabilirim."
    return "TravelMind is currently designed only for hotel and accommodation selection. I do not provide flight, visa, restaurant, or itinerary recommendations. If you share a city and your hotel preferences, I can help with accommodation options."


def no_result_message(lang_code: str) -> str:
    if lang_code == "tr":
        return "İstenen konum için CMU TripAdvisor veri setinde uygun otel sonucu bulunamadı."
    return "No suitable hotel result was found for the requested location in the CMU TripAdvisor dataset."


def empty_query_message(lang_code: str) -> str:
    if lang_code == "tr":
        return "Soru boş olamaz."
    return "The query cannot be empty."


def goodbye_message(lang_code: str) -> str:
    if lang_code == "tr":
        return "Görüşmek üzere! İyi seyahatler."
    return "Goodbye! Have a great trip."


def final_note_lines(lang_code: str):
    if lang_code == "tr":
        return [
            "- Retrieval ve skorlar CMU TripAdvisor datasetinden gelen chunk'lara dayanır.",
            "- Yerel LLM yalnızca bu kanıtları doğal cevaba dönüştürür.",
            "- Fiyat, canlı müsaitlik ve güncel rezervasyon bilgisi üretilmez.",
        ]
    return [
        "- Retrieval and scores are based on chunks from the CMU TripAdvisor dataset.",
        "- The local LLM only turns this evidence into a natural answer.",
        "- Price, live availability, and current booking information are not generated.",
    ]


def ui_title(lang_code: str):
    if lang_code == "tr":
        return "TravelMind Final Cevap"
    return "TravelMind Final Answer"


def initial_welcome_message(lang_code: str) -> str:
    if lang_code == "tr":
        return "TravelMind: Merhaba! Ben TravelMind. Size otel ve konaklama önerilerinde nasıl yardımcı olabilirim? (Çıkış için 'çık' yazabilirsiniz)"
    return "TravelMind: Hello! I'm TravelMind. How can I help you with hotel recommendations? (Type 'exit' to quit)"


def input_prompt(lang_code: str) -> str:
    if lang_code == "tr":
        return "\nSiz: "
    return "\nYou: "

