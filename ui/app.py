import streamlit as st

# 1. EN ÜSTE EKLENMELİ Kİ "STREAMLIT" YAZISI GİTSİN!
st.set_page_config(
    page_title="TravelMind - AI Travel Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import streamlit.components.v1 as components
import sys
import os

# Tarayıcının otomatik çevirisini ENGELLE (Placeholder hatasını çözer)
components.html("""
<script>
    window.parent.document.documentElement.setAttribute("translate", "no");
    window.parent.document.documentElement.classList.add("notranslate");
</script>
""", height=0, width=0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import importlib
import cmu_rag_answer
import travelmind_scoring
import hotel_card_builder
importlib.reload(cmu_rag_answer)
importlib.reload(travelmind_scoring)
importlib.reload(hotel_card_builder)

from cmu_rag_answer import get_llm_intent_and_location, generate_llm_answer, generate_conversational_answer, generate_followup_answer

def sanitize_before_render(answer):
    if not answer or not str(answer).strip():
        return None
        
    sanitized = str(answer)
    forbidden = [
        "Okay, the user",
        "The user is asking",
        "<think>",
        "</think>",
        "Chunk Type:",
        "[Insert evidence summary here]",
        "düşünüyor..."
    ]
    for x in forbidden:
        # Case insensitive replacement
        import re
        sanitized = re.sub(re.escape(x), "", sanitized, flags=re.IGNORECASE)
        
    if not sanitized.strip():
        return None
        
    return sanitized
from cmu_retrieve import search
from travelmind_scoring import calculate_travelmind_score
from answer_validator import validate_answer
from hotel_card_builder import build_hotel_cards


def append_message(msg):
    st.session_state.messages.append(msg)
    import datetime, os
    try:
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'ui_test_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'chat_history.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{ts}] {msg.get('role', 'UNKNOWN').upper()}:\n{msg.get('content', '')}\n' + '-'*40 + '\n')
    except Exception:
        pass

translations = {
    "EN": {
        "flag": "🇺🇸 English", "code": "en",
        "hero_title": "Find your hotel with TravelMind",
        "hero_subtitle": "Search among thousands of hotels in 25 popular U.S. cities to find the one that's right for you.",
        "chat_placeholder": "What kind of hotel are you looking for? (e.g., A luxury hotel with a pool in New York)",
        "analyzing": "TravelMind is analyzing...",
        "exit_msg": "Goodbye! Have a great trip.",
        "unsupported_loc": "Unfortunately, I only serve supported US cities right now.",
        "missing_loc": "Please specify the city you want to search in. (e.g., Boston, Chicago, Miami)",
        "not_found": "Sorry, I couldn't find a hotel matching your criteria in {}.",
        "unknown_hotel": "Unknown Hotel",
        "class_label": "Class",
        "amenities_label": "Amenities",
        "score_label": "TravelMind Score",
        "map": "View on Map",
        "footer": "Bilal Kocakaplan, 2026. All Rights Reserved.",
        "lang_label": "Language",
        "theme_label": "Theme",
        "dark_theme": "🌙 Dark",
        "light_theme": "☀️ Light",
        "lang_warning": "ℹ️ <b>Note:</b> TravelMind answers in the language you select here.<br>⚠️ Changing the language clears unsubmitted text."
    },
    "TR": {
        "flag": "🇹🇷 Türkçe", "code": "tr",
        "hero_title": "Size en uygun oteli TravelMind ile bulun",
        "hero_subtitle": "ABD'nin 25 popüler şehrindeki binlerce otel arasından size en uygun olanı bulmak için arama yapın.",
        "chat_placeholder": "Ne tür bir otel arıyorsunuz? (Örn: New York'ta manzaralı ve havuzlu lüks otel)",
        "analyzing": "TravelMind analiz ediyor...",
        "exit_msg": "Görüşmek üzere! İyi seyahatler dilerim.",
        "unsupported_loc": "Maalesef şu anda sadece desteklenen ABD şehirlerinde hizmet verebiliyorum.",
        "missing_loc": "Lütfen arama yapmak istediğiniz şehri belirtin. (Örn: Boston, Chicago, Miami)",
        "not_found": "Üzgünüm, {} kriterlerinize uygun otel bulamadım.",
        "unknown_hotel": "Bilinmeyen Otel",
        "class_label": "Yıldız Sayısı",
        "amenities_label": "Özellikler",
        "score_label": "TravelMind Puanı",
        "map": "Haritada Gör",
        "footer": "Bilal Kocakaplan, 2026. Tüm Hakları Saklıdır.",
        "lang_label": "Dil",
        "theme_label": "Tema",
        "dark_theme": "🌙 Karanlık",
        "light_theme": "☀️ Aydınlık",
        "lang_warning": "ℹ️ <b>Bilgi:</b> Sistem, seçtiğiniz bu dile göre yanıt verir.<br>⚠️ Dil değişimi sohbet kutusunu sıfırlar."
    },
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_hotel_index" not in st.session_state:
    st.session_state.selected_hotel_index = 0
if "last_hotel_cards" not in st.session_state:
    st.session_state.last_hotel_cards = []
if "theme_internal" not in st.session_state:
    st.session_state.theme_internal = "dark"
if "current_location" not in st.session_state:
    st.session_state.current_location = None
if "language" not in st.session_state:
    st.session_state.language = "EN"

t = translations[st.session_state.language]

if st.session_state.theme_internal == "dark":
    bg_color = "#121212"
    text_color = "#e0e0e0"
    chat_bg = "#1e1e1e"
    chat_text = "#ffffff"
    input_bg = "#2b2b2b"
    input_text = "#ffffff"
    gradient_end = "rgba(18, 18, 18, 1)"
    nav_bg = "rgba(15, 23, 42, 0.98)"
    nav_border = "rgba(255, 255, 255, 0.1)"
    nav_text = "#ffffff"
else:
    bg_color = "#ffffff"
    text_color = "#111111"
    chat_bg = "#0f172a"
    chat_text = "#ffffff"
    input_bg = "#ffffff"
    input_text = "#111111"
    gradient_end = "rgba(255, 255, 255, 1)"
    nav_bg = "rgba(245, 248, 250, 0.98)"
    nav_border = "rgba(0, 0, 0, 0.1)"
    nav_text = "#0f172a"

HERO_BG_URL = "https://images.unsplash.com/photo-1499092346589-b9b6be3e94b2?q=80&w=3840&auto=format&fit=crop"

import base64
import os
logo_path = r"C:\Users\Lenovo\Desktop\travelmind-rag\ui\assets\logo_cropped.png"
logo_base64 = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")

app_bg = f"""
    .stApp {{
        background: linear-gradient(to bottom, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.7) 50%, {gradient_end} 100%), url('{HERO_BG_URL}') no-repeat center center fixed !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        color: {text_color} !important; 
        font-family: 'Inter', sans-serif !important;
    }}
    """

custom_css = f"""
<style>
    {app_bg}
    .custom-footer {{
        text-align: center;
        padding: 20px;
        font-size: 13px;
        color: {text_color} !important;
        opacity: 0.7;
    }}
    /* Chat Input Fixleri */
    div[data-testid="stBottom"], 
    div[data-testid="stBottom"] > div, 
    div[data-testid="stBottomBlockContainer"] {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
    }}
    div[data-testid="stChatInput"] {{
        background-color: transparent !important;
    }}
    div[data-testid="stChatInput"] > div {{
        background-color: {chat_bg} !important;
        border: 1px solid {nav_border} !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: {chat_text} !important;
        background-color: transparent !important;
    }}
    div[data-testid="stChatInput"] button svg {{
        fill: {chat_text} !important;
        color: {chat_text} !important;
    }}
    
    div[data-testid="stChatMessage"] {{
        background-color: {chat_bg} !important;
        border: 1px solid {nav_border} !important;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    div[data-testid="stChatMessage"] * {{
        color: {chat_text} !important;
    }}
    
    /* Selectbox ve Input Fixleri (React Aria ComboBox API'sine özel) */
    div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"] {{
        background-color: {input_bg} !important;
        border: 1px solid {nav_border} !important;
    }}
    div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"] input {{
        color: {input_text} !important;
        background-color: transparent !important;
        pointer-events: none !important;
        cursor: pointer !important;
    }}
    div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"] button svg {{
        fill: {input_text} !important;
        color: {input_text} !important;
    }}

    /* Radio buton text rengi fix */
    div[data-testid="stRadio"] div[role="radiogroup"] p {{
        color: {nav_text} !important;
    }}
    
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden !important;}}
    footer {{visibility: hidden;}}
    
    .block-container {{
        padding-top: 100px !important;
        padding-bottom: 5rem !important;
        max-width: 900px !important;
    }}

    /* Kusursuz Tam Genişlik Navbar Görünümü (Koyu Lacivert Tema) */
    div[data-testid="stHorizontalBlock"]:first-of-type:not(div[data-testid="stChatMessage"] *) {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        max-width: none;
        background-color: {nav_bg};
        backdrop-filter: blur(15px);
        padding: 15px 5vw;
        border-radius: 0;
        border-bottom: 1px solid {nav_border};
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.8);
        z-index: 999999;
        align-items: center;
    }}
    
    /* Navbar içi elementleri hizala */
    div[data-testid="stHorizontalBlock"]:first-of-type:not(div[data-testid="stChatMessage"] *) * {{
        color: {nav_text} !important;
    }}
    
    /* Şık kutucuk tasarımları için Navbar Label stili (Sadece Ana Başlıklar) */
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stWidgetLabel"] p {{
        background: rgba(128, 128, 128, 0.2) !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        display: inline-block !important;
        margin-bottom: 5px !important;
        color: {nav_text} !important;
    }}

    .hero-container {{
        text-align: center;
        margin-top: 10vh;
        margin-bottom: 50px;
    }}
    .hero-title {{
        font-size: 4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 15px;
        text-shadow: 0px 4px 15px rgba(0,0,0,0.6);
        letter-spacing: -1px;
    }}
    .hero-subtitle {{
        font-size: 1.4rem;
        font-weight: 400;
        color: #f0f0f0;
        text-shadow: 0px 2px 10px rgba(0,0,0,0.8);
    }}
    
    .custom-footer {{
        text-align: center;
        padding: 15px;
        font-size: 12px;
        margin-top: 30px;
        color: {text_color} !important;
        opacity: 0.7;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ÜST NAVBAR - Kusursuz Tam Genişlik ve Logo
nav_col1, nav_space, nav_col2, nav_col3 = st.columns([3, 4, 1.5, 1.5])

with nav_col1:
    if logo_base64:
        st.markdown(f'<img src="data:image/png;base64,{logo_base64}" style="height: 80px; margin-top: 5px; object-fit: contain;">', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size: 26px; font-weight: 900; color: #ffffff; letter-spacing: 1px; margin-top: 4px;">TravelMind</div>', unsafe_allow_html=True)

def on_lang_change():
    st.session_state.language = st.session_state.lang_selector

with nav_col2:
    st.radio(
        t["lang_label"],
        options=list(translations.keys()),
        format_func=lambda x: translations[x]["flag"],
        index=list(translations.keys()).index(st.session_state.language),
        key="lang_selector",
        horizontal=True,
        on_change=on_lang_change,
        label_visibility="collapsed"
    )
    st.markdown(f"<div style='font-size: 11px; color: #ffcccc; margin-top: -10px; margin-bottom: 10px;'>{t['lang_warning']}</div>", unsafe_allow_html=True)

def on_theme_change():
    if st.session_state.theme_selector == t["dark_theme"]:
        st.session_state.theme_internal = "dark"
    else:
        st.session_state.theme_internal = "light"

with nav_col3:
    st.radio(
        t["theme_label"], 
        [t["dark_theme"], t["light_theme"]], 
        index=0 if st.session_state.theme_internal == "dark" else 1,
        key="theme_selector",
        horizontal=True,
        on_change=on_theme_change,
        label_visibility="collapsed"
    )


for msg in st.session_state.messages:
    avatar = "🌍" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "hotels" in msg and msg["hotels"]:
            for card in msg["hotels"]:
                hotel_name = card.get('hotel_name', t['unknown_hotel'])
                location = card.get('location', '')
                h_class = card.get('hotel_class', '')
                if not h_class or str(h_class).lower() in ["nan", "none", "unknown", "null", ""]:
                    h_class = "Belirtilmemiş" if t['code'] == 'tr' else "Not Specified"
                
                score = card.get('travelmind_score', 0)
                map_url = card.get('map_link')
                phone_number = card.get('phone', 'UNKNOWN')
                
                amenities_dict = card.get('amenities', {})
                if isinstance(amenities_dict, list):
                    amenities_dict = {} 
                
                nice_am = []
                for k, v in amenities_dict.items():
                    if k == 'other':
                        if isinstance(v, list):
                            for o_am in v[:4]:
                                nice_am.append(str(o_am).title())
                        continue
                    if v == 'YES':
                        if k == 'wifi': nice_am.append("Wi-Fi" if t['code'] != 'tr' else "Wi-Fi (İnternet)")
                        elif k == 'pool': nice_am.append("Pool" if t['code'] != 'tr' else "Havuz")
                        elif k == 'gym' or k == 'gym_fitness': nice_am.append("Gym / Fitness" if t['code'] != 'tr' else "Spor Salonu")
                        elif k == 'breakfast': nice_am.append("Breakfast" if t['code'] != 'tr' else "Kahvaltı")
                        elif k == 'parking': nice_am.append("Parking" if t['code'] != 'tr' else "Otopark")
                        elif k == 'wheelchair_accessible': nice_am.append("Wheelchair Accessible" if t['code'] != 'tr' else "Engelli Erişimi")
                        elif k == 'pet_friendly': nice_am.append("Pet Friendly" if t['code'] != 'tr' else "Evcil Hayvan Dostu")
                        else: nice_am.append(k.replace('_', ' ').capitalize())
                
                room_info_dict = card.get('room_info', {})
                room_types_list = room_info_dict.get('room_types', room_info_dict.get('booking_room_types', []))
                if isinstance(room_types_list, str): room_types_list = [room_types_list]
                
                has_suite = room_info_dict.get("suite") == "YES"
                rooms_list = []
                if has_suite:
                    rooms_list.append("Suite (Suit Oda)" if t['code'] == 'tr' else "Suite")
                if room_types_list:
                    for rt in room_types_list[:4]:
                        if "suite" not in str(rt).lower():
                            rooms_list.append(str(rt))
                            
                with st.container():
                    st.markdown("---")
                    st.markdown(f"#### 🏨 {hotel_name}")
                    c1, c2 = st.columns([2.5, 1.5])
                    with c1:
                        st.write(f"📍 **{location}** | {t['class_label']}: {h_class}")
                        if phone_number and phone_number != 'UNKNOWN' and str(phone_number).lower() not in ['none', 'null']:
                            st.write(f"📞 **Telefon:** {phone_number}" if t['code'] == 'tr' else f"📞 **Phone:** {phone_number}")
                        if nice_am:
                            if t['code'] == 'tr':
                                msg_am = "Misafirlerimize sunulan ayrıcalıklar arasında; " + ", ".join(nice_am) + " bulunmaktadır."
                            else:
                                msg_am = "Exclusive privileges offered to our guests include " + ", ".join(nice_am) + "."
                            st.write(f"🛠️ **{t['amenities_label']}:** {msg_am}")
                        else:
                            empty_msg = "Detaylı olanak bilgisi mevcut kayıtlarımızda teyit edilememiştir." if t['code'] == 'tr' else "Detailed amenity information could not be verified in our current records."
                            st.write(f"🛠️ **{t['amenities_label']}:** {empty_msg}")
                            
                        if rooms_list:
                            if t['code'] == 'tr':
                                room_msg = ", ".join(rooms_list) + " vb. oda seçenekleri mevcuttur."
                            else:
                                room_msg = ", ".join(rooms_list) + " and similar room options are available."
                            st.write(f"🛏️ **Oda Tipleri:** {room_msg}" if t['code'] == 'tr' else f"🛏️ **Room Types:** {room_msg}")
                            
                        req_warnings = card.get("requirement_satisfaction", {})
                        if req_warnings:
                            for req_key, req_status in req_warnings.items():
                                if req_status == "UNKNOWN" or req_status == "MISSING":
                                    warn_msg = f"{req_key.replace('_', ' ').capitalize()}: Cannot be confirmed from the current dataset." if t['code'] != 'tr' else f"{req_key.replace('_', ' ').capitalize()}: Mevcut veri setinde doğrulanamadı."
                                    st.write(f"⚠️ **{warn_msg}**")
                                    
                    with c2:
                        st.metric(t["score_label"], f"{score:.1f}/100")
                        if map_url and map_url != "UNKNOWN":
                            map_label = "Haritada Gör" if t['code'] == 'tr' else "View on Map"
                            st.markdown(f"👉 **[{map_label}]({map_url})**", unsafe_allow_html=True)

if len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div class="hero-container" translate="no">
        <div class="hero-title">{t["hero_title"]}</div>
        <div class="hero-subtitle">{t["hero_subtitle"]}</div>
    </div>
    """, unsafe_allow_html=True)

if prompt := st.chat_input(t["chat_placeholder"]):
    append_message({"role": "user", "content": prompt})
    st.rerun()
    
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant", avatar="🌍"):
        with st.spinner(t["analyzing"]):
            from cmu_rag_answer import fast_route_query
            fast_res = fast_route_query(prompt, st.session_state)
            
            if fast_res:
                intent = fast_res.get("intent")
                parsed_location = fast_res.get("location", fast_res.get("city", None))
                query_requirements = fast_res.get("query_requirements", fast_res.get("requirements", {}))
            else:
                router_res = get_llm_intent_and_location(prompt, st.session_state.messages[:-1])
                intent = router_res.get("intent", "general_chat")
                parsed_location = router_res.get("location", None)
                query_requirements = router_res.get("query_requirements", {})

            if parsed_location:
                st.session_state.current_location = parsed_location
            
            effective_location = st.session_state.current_location

            if intent == "exit":
                response_text = t["exit_msg"]
                st.markdown(response_text)
                append_message({"role": "assistant", "content": response_text})
                st.rerun()
            
            elif intent == "unsupported_location":
                answer_container = st.empty()
                city_name = parsed_location if parsed_location else "bu şehir"
                
                if t["code"] == 'tr':
                    full_answer = f"TravelMind şu anda {city_name} için otel önerisi sunmuyor. Sistemimiz yalnızca desteklenen şehirlerdeki oteller için çalışır. Fiyat, canlı müsaitlik veya rezervasyon bilgisi sağlamaz."
                else:
                    full_answer = f"TravelMind currently does not support hotel recommendations for {city_name}. Our system only works for supported cities. It does not provide live prices, availability, or booking information."
                
                answer_container.markdown(full_answer, unsafe_allow_html=True)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()
                
            elif intent == "missing_location" and not effective_location:
                response_text = t["missing_loc"]
                st.markdown(response_text)
                append_message({"role": "assistant", "content": response_text})
                st.rerun()
                
            elif intent == "out_of_scope":
                answer_container = st.empty()
                if True:
                    fallback_msg = {
                        "en": "I am a hotel recommendation assistant. I cannot answer questions outside of hotel search and travel accommodations.",
                        "tr": "Ben bir otel tavsiye asistanıyım. Otel aramaları ve konaklama dışındaki sorulara yanıt veremiyorum."
                    }
                    full_answer = fallback_msg.get(t["code"], fallback_msg["tr"])
                            
                answer_container.markdown(full_answer, unsafe_allow_html=True)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()
                
            elif intent == "class_explanation":
                answer_container = st.empty()
                if t["code"] == 'tr':
                    full_answer = (
                        "Otel Sınıfı (Class) doğrudan Tripadvisor veri setinden alınan resmi bir meta veridir ve "
                        "sistemimiz tarafından hesaplanmaz. Sadece otelin kayıtlı olan resmi yıldızını (örn. 4.0, 5.0) "
                        "temsil eder. Ancak bu değer, nihai TravelMind uygunluk skorunuz hesaplanırken %35 oranında bir etkiye sahiptir."
                    )
                else:
                    full_answer = (
                        "The Hotel Class is an official metadata value directly pulled from the Tripadvisor dataset "
                        "and is not calculated by our system. It simply represents the hotel's registered official "
                        "star rating (e.g., 4.0, 5.0). However, this value accounts for 35% of the final TravelMind suitability score."
                    )
                
                answer_container.markdown(full_answer, unsafe_allow_html=True)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()
                
            elif intent == "score_explanation":
                answer_container = st.empty()
                if t["code"] == 'tr':
                    full_answer = (
                        "TravelMind uygunluk skoru, sistem kayıtlarımızdaki zenginleştirilmiş verilere dayanarak hesaplanır. "
                        "Puanlama ağırlıklı olarak şunları içerir: Otel yıldız sınıfı (%35), Konum eşleşmesi (%25), "
                        "Sunulan olanakların zenginliği (Wi-Fi, Havuz vb.) (%15), Yatak tipi eşleşmesi (%15), "
                        "Toplam yorum sayısı güven sinyali (%5) ve Ziyaretçi yorumlarındaki temizlik/kalite hissiyatı (%5). "
                        "Bu sayede size veri destekli, güvenilir ve tarafsız bir sıralama sunulur."
                    )
                else:
                    full_answer = (
                        "The TravelMind suitability score is calculated based on the enriched metadata in our system. "
                        "The scoring primarily includes: Hotel class/stars (35%), Location match (25%), "
                        "Richness of amenities like Wi-Fi and Pool (15%), Bed type match (15%), "
                        "Total review count as a trust signal (5%), and Cleanliness/quality sentiment from visitor reviews (5%). "
                        "This ensures you receive a data-backed, reliable, and unbiased ranking."
                    )
                
                answer_container.markdown(full_answer, unsafe_allow_html=True)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()
                
            elif intent == "general_chat":
                answer_container = st.empty()
                if True:
                    generator = generate_conversational_answer(prompt, t["code"], st.session_state.messages[:-1])
                    full_answer = ""
                    for chunk in generator:
                        if isinstance(chunk, dict) and chunk["type"] == "answer":
                            full_answer += chunk["content"]
                        elif not isinstance(chunk, dict):
                            full_answer += chunk
                            
                    full_answer = sanitize_before_render(full_answer)
                    if full_answer is None:
                        fallback_msg = {
                            "en": "I am a hotel recommendation assistant. How can I help you with your travel plans today?",
                            "tr": "Ben bir otel tavsiye asistanıyım. Bugün seyahat planlarınızda size nasıl yardımcı olabilirim?",
                            "de": "Ich bin ein Hotel-Empfehlungsassistent. Wie kann ich Ihnen heute bei Ihren Reiseplänen helfen?",
                            "fr": "Je suis un assistant de recommandation d'hôtels. Comment puis-je vous aider avec vos projets de voyage aujourd'hui?",
                            "it": "Sono un assistente per le raccomandazioni alberghiere. Come posso aiutarti con i tuoi programmi di viaggio oggi?",
                            "zh": "我是酒店推荐助手。今天我能为您的旅行计划做些什么？"
                        }
                        full_answer = fallback_msg.get(t["code"], fallback_msg["tr"])
                answer_container.markdown(full_answer, unsafe_allow_html=True)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()
                
            elif intent in ["price_question", "followup_pool", "followup_breakfast", "followup_other_hotel", "follow_up", "specific_hotel_info"]:
                last_cards = st.session_state.get("last_hotel_cards", [])
                selected_idx = st.session_state.get("selected_hotel_index", 0)
                answer_container = st.empty()
                
                if intent == "price_question":
                    if t["code"] == 'tr':
                        response_text = "TravelMind, sistemimizde gerçek zamanlı rezervasyon verisi bulunmadığı için canlı fiyat veya müsaitlik bilgisi sağlamaz."
                    else:
                        response_text = "TravelMind does not provide live pricing or availability because our system does not contain real-time booking data."
                    answer_container.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()
                
                elif intent == "followup_pool":
                    if not last_cards:
                        response_text = "I do not have previous hotel results to check for a pool. Please search for a city first." if t["code"] != "tr" else "Havuz durumunu kontrol etmek için önceki bir otel sonucu yok. Lütfen önce bir şehir araması yapın."
                    else:
                        current_card = last_cards[selected_idx]
                        pool_status = current_card.get("amenities", {}).get("pool", "UNKNOWN")
                        if pool_status == "YES":
                            response_text = "The selected hotel has pool information confirmed in our records." if t["code"] != "tr" else "Seçilen otelin havuzu olduğu sistem kayıtlarımızda doğrulanmıştır."
                        elif pool_status == "NO":
                            response_text = "Pool is not listed in our records for this hotel." if t["code"] != "tr" else "Sistem kayıtlarımızda bu otel için havuz görünmüyor."
                        else:
                            response_text = "Pool information is not clearly confirmed in our system for this hotel." if t["code"] != "tr" else "Bu otel için havuz bilgisi sistem kayıtlarımızda net olarak doğrulanmamıştır."
                    
                    answer_container.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()
                    
                elif intent == "followup_breakfast":
                    if not last_cards:
                        response_text = "I do not have previous hotel results to check for breakfast. Please search for a city first." if t["code"] != "tr" else "Kahvaltı durumunu kontrol etmek için önceki bir otel sonucu yok. Lütfen önce bir şehir araması yapın."
                    else:
                        current_card = last_cards[selected_idx]
                        bf_status = current_card.get("amenities", {}).get("breakfast", "UNKNOWN")
                        if bf_status == "YES":
                            response_text = "The selected hotel has breakfast confirmed in our records." if t["code"] != "tr" else "Seçilen otelin kahvaltı hizmeti olduğu sistem kayıtlarımızda doğrulanmıştır."
                        elif bf_status == "NO":
                            response_text = "Breakfast is not listed in our records for this hotel." if t["code"] != "tr" else "Sistem kayıtlarımızda bu otel için kahvaltı görünmüyor."
                        else:
                            response_text = "Breakfast information is not clearly confirmed in our system for this hotel." if t["code"] != "tr" else "Bu otel için kahvaltı bilgisi sistem kayıtlarımızda net olarak doğrulanmamıştır."
                    
                    answer_container.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()

                elif intent == "followup_other_hotel":
                    if not last_cards:
                        response_text = "I do not have previous hotel results to choose another hotel from. Please search for a city first." if t["code"] != "tr" else "Başka bir otel seçebileceğim önceki bir arama sonucu yok. Lütfen önce bir şehir araması yapın."
                    else:
                        new_idx = selected_idx + 1
                        if new_idx >= len(last_cards):
                            response_text = "I have shown all the hotels from the current search results. Please refine your search." if t["code"] != "tr" else "Mevcut arama sonuçlarındaki tüm otelleri gösterdim. Lütfen aramanızı daraltın."
                        else:
                            st.session_state.selected_hotel_index = new_idx
                            current_card = last_cards[new_idx]
                            hotel_name = current_card.get('hotel_name', 'UNKNOWN')
                            score = current_card.get('travelmind_score', 0)
                            response_text = f"Sure, the next hotel is **{hotel_name}**. It has a TravelMind score of {score:.1f}/100." if t["code"] != "tr" else f"Tabii, sıradaki otel **{hotel_name}**. Bu otelin TravelMind puanı {score:.1f}/100."
                    
                    answer_container.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()
                
                else:
                    # Generic follow-up logic using LLM
                    from cmu_rag_answer import build_hotel_context
                    
                    req_hotel = None
                    if 'router_res' in locals() and router_res:
                        req_hotel = router_res.get("requested_hotel_name")
                    elif 'fast_res' in locals() and fast_res:
                        req_hotel = fast_res.get("requested_hotel_name")
                        
                    cards_to_include = last_cards
                    if req_hotel:
                        matched = [c for c in last_cards if req_hotel.lower() in c.get('hotel_name', '').lower()]
                        if matched:
                            cards_to_include = matched
                    
                    # If router didn't catch the exact name, check if prompt contains any hotel name from last_cards
                    if len(cards_to_include) == len(last_cards):
                        prompt_lower = prompt.lower()
                        matched_from_prompt = [c for c in last_cards if c.get('hotel_name', '').lower() in prompt_lower]
                        if matched_from_prompt:
                            # If multiple match, we take the first one or all matched. All matched is safer.
                            cards_to_include = matched_from_prompt
                            
                    context_str = ""
                    for i, card in enumerate(cards_to_include, start=1):
                        context_str += build_hotel_context(prompt, card, i) + "\n\n"
                        
                    generator = generate_followup_answer(
                        query=prompt, 
                        context_str=context_str, 
                        lang_code=t["code"], 
                        chat_history=st.session_state.messages[:-1]
                    )
                    full_answer = ""
                    for chunk in generator:
                        if isinstance(chunk, dict) and chunk["type"] == "answer":
                            full_answer += chunk["content"]
                        elif not isinstance(chunk, dict):
                            full_answer += chunk
                                
                    from answer_validator import validate_answer
                    validation = validate_answer(full_answer, last_cards, "follow_up", None, t["code"])
                    final_display = validation["sanitized_answer"]
                    
                    final_display = sanitize_before_render(final_display)
                    if final_display is None:
                        from cmu_rag_answer import safe_card_based_fallback_answer
                        final_display = safe_card_based_fallback_answer(
                            user_query=prompt,
                            hotel_cards=cards_to_include,
                            language=t["code"]
                        )
                    
                    answer_container.markdown(final_display, unsafe_allow_html=True)
                    append_message({"role": "assistant", "content": final_display})
                    st.rerun()

            else:
                results = search(prompt, location_filter=effective_location, filters=None, top_k_hotels=12)
                
                if not results:
                    response_text = t["not_found"].format(effective_location or '')
                    st.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()
                else:
                    cards_json = build_hotel_cards(
                        results=results,
                        user_query=prompt,
                        query_requirements=query_requirements,
                        requested_location=locals().get("effective_location"),
                        effective_scores=locals().get("effective_scores"),
                    )
                    sorted_cards = sorted(cards_json, key=lambda x: x.get("rank_score", 0.0), reverse=True)[:4]
                    
                    answer_container = st.empty()
                    
                    # Generate full answer silently with spinner
                    if True:
                        from cmu_rag_answer import build_hotel_context
                        hotel_context_str = ""
                        for i, result in enumerate(sorted_cards, start=1):
                            hotel_context_str += build_hotel_context(prompt, result, i) + "\n\n"
                            
                        generator = generate_llm_answer(
                            query=prompt, 
                            hotel_context_str=hotel_context_str, 
                            chat_history=st.session_state.messages[:-1], 
                            location=effective_location, 
                            lang_code=t["code"],
                            hotel_cards=sorted_cards
                        )
                        full_answer = ""
                        for chunk in generator:
                            if isinstance(chunk, dict) and chunk["type"] == "answer":
                                full_answer += chunk["content"]
                            elif not isinstance(chunk, dict):
                                full_answer += chunk
                                
                        validation = validate_answer(full_answer, sorted_cards, "hotel_search", effective_location, t["code"])
                        final_display = validation["sanitized_answer"]
                        
                        final_display = sanitize_before_render(final_display)
                        if final_display is None:
                            from cmu_rag_answer import safe_card_based_fallback_answer
                            final_display = safe_card_based_fallback_answer(
                                user_query=prompt,
                                hotel_cards=sorted_cards,
                                query_requirements=query_requirements,
                                city=effective_location,
                                language=t["code"]
                            )
                        
                    # Update session state with the new context immediately
                    st.session_state["last_hotel_cards"] = sorted_cards
                    st.session_state["selected_hotel_index"] = 0
                    if effective_location:
                        st.session_state["last_search_city"] = effective_location
                    if sorted_cards:
                        st.session_state["last_answered_hotel_name"] = sorted_cards[0].get("hotel_name", "UNKNOWN")
                        
                    print(f"[STATE] last_hotel_cards_count={len(sorted_cards)} selected_hotel_index=0 last_search_city={effective_location}")

                    answer_container.markdown(final_display, unsafe_allow_html=True)
                    st.markdown("---")
                    
                    for card in sorted_cards:
                        hotel_name = card.get('hotel_name', t['unknown_hotel'])
                        location = card.get('location', '')
                        h_class = card.get('hotel_class', '')
                        if not h_class or str(h_class).lower() in ["nan", "none", "unknown", "null", ""]:
                            h_class = "Belirtilmemiş" if t['code'] == 'tr' else "Not Specified"
                        
                        score = card.get('travelmind_score', 0)
                        map_url = card.get('map_link')
                        phone_number = card.get('phone', 'UNKNOWN')
                        
                        amenities_dict = card.get('amenities', {})
                        if isinstance(amenities_dict, list):
                            amenities_dict = {} 
                        
                        nice_am = []
                        for k, v in amenities_dict.items():
                            if k == 'other':
                                if isinstance(v, list):
                                    for o_am in v[:4]:
                                        nice_am.append(str(o_am).title())
                                continue
                            if v == 'YES':
                                if k == 'wifi': nice_am.append("Wi-Fi" if t['code'] != 'tr' else "Wi-Fi (İnternet)")
                                elif k == 'pool': nice_am.append("Pool" if t['code'] != 'tr' else "Havuz")
                                elif k == 'gym' or k == 'gym_fitness': nice_am.append("Gym / Fitness" if t['code'] != 'tr' else "Spor Salonu")
                                elif k == 'breakfast': nice_am.append("Breakfast" if t['code'] != 'tr' else "Kahvaltı")
                                elif k == 'parking': nice_am.append("Parking" if t['code'] != 'tr' else "Otopark")
                                elif k == 'wheelchair_accessible': nice_am.append("Wheelchair Accessible" if t['code'] != 'tr' else "Engelli Erişimi")
                                elif k == 'pet_friendly': nice_am.append("Pet Friendly" if t['code'] != 'tr' else "Evcil Hayvan Dostu")
                                else: nice_am.append(k.replace('_', ' ').capitalize())
                        
                        room_info_dict = card.get('room_info', {})
                        room_types_list = room_info_dict.get('room_types', room_info_dict.get('booking_room_types', []))
                        if isinstance(room_types_list, str): room_types_list = [room_types_list]
                        
                        has_suite = room_info_dict.get("suite") == "YES"
                        rooms_list = []
                        if has_suite:
                            rooms_list.append("Suite (Suit Oda)" if t['code'] == 'tr' else "Suite")
                        if room_types_list:
                            for rt in room_types_list[:4]:
                                if "suite" not in str(rt).lower():
                                    rooms_list.append(str(rt))
                                    
                        with st.container():
                            st.markdown("---")
                            st.markdown(f"#### 🏨 {hotel_name}")
                            c1, c2 = st.columns([2.5, 1.5])
                            with c1:
                                st.write(f"📍 **{location}** | {t['class_label']}: {h_class}")
                                if phone_number and phone_number != 'UNKNOWN' and str(phone_number).lower() not in ['none', 'null']:
                                    st.write(f"📞 **Telefon:** {phone_number}" if t['code'] == 'tr' else f"📞 **Phone:** {phone_number}")
                                if nice_am:
                                    if t['code'] == 'tr':
                                        msg_am = "Misafirlerimize sunulan ayrıcalıklar arasında; " + ", ".join(nice_am) + " bulunmaktadır."
                                    else:
                                        msg_am = "Exclusive privileges offered to our guests include " + ", ".join(nice_am) + "."
                                    st.write(f"🛠️ **{t['amenities_label']}:** {msg_am}")
                                else:
                                    empty_msg = "Detaylı olanak bilgisi mevcut kayıtlarımızda teyit edilememiştir." if t['code'] == 'tr' else "Detailed amenity information could not be verified in our current records."
                                    st.write(f"🛠️ **{t['amenities_label']}:** {empty_msg}")
                                    
                                if rooms_list:
                                    if t['code'] == 'tr':
                                        room_msg = ", ".join(rooms_list) + " vb. oda seçenekleri mevcuttur."
                                    else:
                                        room_msg = ", ".join(rooms_list) + " and similar room options are available."
                                    st.write(f"🛏️ **Oda Tipleri:** {room_msg}" if t['code'] == 'tr' else f"🛏️ **Room Types:** {room_msg}")
                                    
                                req_warnings = card.get("requirement_satisfaction", {})
                                if req_warnings:
                                    for req_key, req_status in req_warnings.items():
                                        if req_status == "UNKNOWN" or req_status == "MISSING":
                                            warn_msg = f"{req_key.replace('_', ' ').capitalize()}: Cannot be confirmed from the current dataset." if t['code'] != 'tr' else f"{req_key.replace('_', ' ').capitalize()}: Mevcut veri setinde doğrulanamadı."
                                            st.write(f"⚠️ **{warn_msg}**")
                                            
                            with c2:
                                st.metric(t["score_label"], f"{score:.1f}/100")
                                if map_url and map_url != "UNKNOWN":
                                    map_label = "Haritada Gör" if t['code'] == 'tr' else "View on Map"
                                    st.markdown(f"👉 **[{map_label}]({map_url})**", unsafe_allow_html=True)
                        st.markdown("---")
                    
                    st.session_state.last_hotel_cards = sorted_cards
                    st.session_state.selected_hotel_index = 0
                    
                    append_message({
                        "role": "assistant",
                        "content": final_display,
                        "hotels": sorted_cards
                    })
                    st.rerun()


st.markdown(f"<div class='custom-footer'>{t['footer']}</div>", unsafe_allow_html=True)
