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

from cmu_rag_answer import get_llm_intent_and_location, generate_llm_answer, generate_conversational_answer, generate_followup_answer
from cmu_retrieve import search
from cmu_recommend_hotels import calculate_recommendation_score, build_strengths, build_cautions

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
        "book_now": "Book Now",
        "footer": "Bilal Kocakaplan, 2026. All Rights Reserved.",
        "lang_label": "Language",
        "theme_label": "Theme",
        "dark_theme": "🌙 Dark",
        "light_theme": "☀️ Light"
    },
    "TR": {
        "flag": "🇹🇷 Türkçe", "code": "tr",
        "hero_title": "Amerika'yı Yapay Zeka ile Keşfedin",
        "hero_subtitle": "ABD'nin 25 popüler şehrindeki on binlerce otel arasından size en uygun olanı bulmak için doğal dilde arama yapın.",
        "chat_placeholder": "Ne tür bir otel arıyorsunuz? (Örn: New York'ta manzaralı ve havuzlu lüks otel)",
        "analyzing": "TravelMind analiz ediyor...",
        "exit_msg": "Görüşmek üzere! İyi seyahatler dilerim.",
        "unsupported_loc": "Maalesef şu anda sadece desteklenen ABD şehirlerinde hizmet verebiliyorum.",
        "missing_loc": "Lütfen arama yapmak istediğiniz şehri belirtin. (Örn: Boston, Chicago, Miami)",
        "not_found": "Üzgünüm, {} kriterlerinize uygun otel bulamadım.",
        "unknown_hotel": "Bilinmeyen Otel",
        "class_label": "Sınıf",
        "amenities_label": "Özellikler",
        "score_label": "TravelMind Puanı",
        "book_now": "Hemen İncele",
        "footer": "Bilal Kocakaplan, 2026. Tüm Hakları Saklıdır.",
        "lang_label": "Dil",
        "theme_label": "Tema",
        "dark_theme": "🌙 Karanlık",
        "light_theme": "☀️ Aydınlık"
    },
    "DE": {
        "flag": "🇩🇪 Deutsch", "code": "de",
        "hero_title": "Finden Sie Ihr Hotel mit TravelMind",
        "hero_subtitle": "Suchen Sie unter Tausenden von Hotels in 25 beliebten US-Städten, um das für Sie passende zu finden.",
        "chat_placeholder": "Welche Art von Hotel suchen Sie? (z.B. Ein Luxushotel mit Pool in New York)",
        "analyzing": "TravelMind analysiert...",
        "exit_msg": "Auf Wiedersehen! Wir wünschen Ihnen eine gute Reise.",
        "unsupported_loc": "Leider bediene ich derzeit nur unterstützte US-Städte.",
        "missing_loc": "Bitte geben Sie die Stadt an, in der Sie suchen möchten. (z. B. Boston, Chicago, Miami)",
        "not_found": "Es tut uns leid, ich konnte in {} kein Hotel finden, das Ihren Kriterien entspricht.",
        "unknown_hotel": "Unbekanntes Hotel",
        "class_label": "Klasse",
        "amenities_label": "Ausstattung",
        "score_label": "TravelMind-Ergebnis",
        "book_now": "Jetzt buchen",
        "footer": "Bilal Kocakaplan, 2026. Alle Rechte vorbehalten.",
        "lang_label": "Sprache",
        "theme_label": "Thema",
        "dark_theme": "🌙 Dunkel",
        "light_theme": "☀️ Hell"
    },
    "FR": {
        "flag": "🇫🇷 Français", "code": "fr",
        "hero_title": "Trouvez votre hôtel avec TravelMind",
        "hero_subtitle": "Recherchez parmi des milliers d'hôtels dans 25 villes américaines populaires pour trouver celui qui vous convient.",
        "chat_placeholder": "Quel genre d'hôtel recherchez-vous ? (ex. Un hôtel de luxe avec piscine à New York)",
        "analyzing": "TravelMind analyse...",
        "exit_msg": "Au revoir! Bon voyage.",
        "unsupported_loc": "Malheureusement, je ne dessers actuellement que les villes américaines prises en charge.",
        "missing_loc": "Veuillez préciser la ville dans laquelle vous souhaitez effectuer la recherche. (ex. Boston, Chicago, Miami)",
        "not_found": "Désolé, je n'ai pas trouvé d'hôtel correspondant à vos critères à {}.",
        "unknown_hotel": "Hôtel inconnu",
        "class_label": "Classe",
        "amenities_label": "Équipements",
        "score_label": "Score TravelMind",
        "book_now": "Réservez maintenant",
        "footer": "Bilal Kocakaplan, 2026. Tous droits réservés.",
        "lang_label": "Langue",
        "theme_label": "Thème",
        "dark_theme": "🌙 Sombre",
        "light_theme": "☀️ Clair"
    },
    "IT": {
        "flag": "🇮🇹 Italiano", "code": "it",
        "hero_title": "Trova il tuo hotel con TravelMind",
        "hero_subtitle": "Cerca tra migliaia di hotel in 25 famose città degli Stati Uniti per trovare quello giusto per te.",
        "chat_placeholder": "Che tipo di hotel stai cercando? (es. Un hotel di lusso con piscina a New York)",
        "analyzing": "TravelMind sta analizzando...",
        "exit_msg": "Arrivederci! Buon viaggio.",
        "unsupported_loc": "Sfortunatamente, al momento servo solo le città degli Stati Uniti supportate.",
        "missing_loc": "Specifica la città in cui desideri effettuare la ricerca. (es. Boston, Chicago, Miami)",
        "not_found": "Spiacenti, non sono riuscito a trovare un hotel corrispondente ai tuoi criteri in {}.",
        "unknown_hotel": "Hotel sconosciuto",
        "class_label": "Classe",
        "amenities_label": "Servizi",
        "score_label": "Punteggio TravelMind",
        "book_now": "Prenota ora",
        "footer": "Bilal Kocakaplan, 2026. Tutti i diritti riservati.",
        "lang_label": "Lingua",
        "theme_label": "Tema",
        "dark_theme": "🌙 Scuro",
        "light_theme": "☀️ Chiaro"
    },
    "ZH": {
        "flag": "🇨🇳 中文", "code": "zh",
        "hero_title": "与 TravelMind 一起寻找您的酒店",
        "hero_subtitle": "在美国 25 个热门城市的数千家酒店中搜索，找到最适合您的酒店。",
        "chat_placeholder": "您在寻找什么样的酒店？（例如：纽约带游泳池的豪华酒店）",
        "analyzing": "TravelMind 正在分析...",
        "exit_msg": "再见！祝您旅途愉快。",
        "unsupported_loc": "很抱歉，我目前只服务于受支持的美国城市。",
        "missing_loc": "请指定您要搜索的城市。（例如：波士顿，芝加哥，迈阿密）",
        "not_found": "抱歉，我无法在 {} 找到符合您标准的酒店。",
        "unknown_hotel": "未知酒店",
        "class_label": "星级",
        "amenities_label": "设施",
        "score_label": "TravelMind 评分",
        "book_now": "立即查看",
        "footer": "Bilal Kocakaplan, 2026. 保留所有权利。",
        "lang_label": "语言",
        "theme_label": "主题",
        "dark_theme": "🌙 暗色",
        "light_theme": "☀️ 亮色"
    }
}

if "messages" not in st.session_state:
    st.session_state.messages = []
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
    
    /* Selectbox ve Input Fixleri (React Aria ComboBox API'sine özel) */
    div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"] {{
        background-color: {input_bg} !important;
        border: 1px solid {nav_border} !important;
    }}
    div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"] input {{
        color: {input_text} !important;
        background-color: transparent !important;
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

with nav_col2:
    selected_lang_name = st.selectbox(
        t["lang_label"],
        options=list(translations.keys()),
        format_func=lambda x: translations[x]["flag"],
        index=list(translations.keys()).index(st.session_state.language)
    )
    if selected_lang_name != st.session_state.language:
        st.session_state.language = selected_lang_name
        st.rerun()

with nav_col3:
    selected_theme = st.radio(
        t["theme_label"], 
        [t["dark_theme"], t["light_theme"]], 
        horizontal=True,
        index=0 if st.session_state.theme_internal == "dark" else 1
    )
    if selected_theme == t["dark_theme"]:
        if st.session_state.theme_internal != "dark":
            st.session_state.theme_internal = "dark"
            st.rerun()
    else:
        if st.session_state.theme_internal != "light":
            st.session_state.theme_internal = "light"
            st.rerun()


for msg in st.session_state.messages:
    avatar = "🌍" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "hotels" in msg and msg["hotels"]:
            for res in msg["hotels"]:
                meta = res["metadata"]
                score_data = calculate_recommendation_score(res)
                final_score = score_data["score"]
                
                with st.container():
                    st.markdown("---")
                    st.markdown(f"#### 🏨 {meta.get('hotel_name', t['unknown_hotel'])}")
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(f"📍 **{meta.get('location', '')}** | {t['class_label']}: {meta.get('hotel_class', '')}")
                        st.write(f"🛠️ **{t['amenities_label']}:** {', '.join(meta.get('amenities', [])[:6])}")
                    with c2:
                        st.metric(t["score_label"], f"%{final_score:.1f}")
                        booking_url = f"https://www.booking.com/searchresults.html?ss={meta.get('hotel_name', '').replace(' ', '+')}"
                        st.markdown(f"👉 **[{t['book_now']}]({booking_url})**", unsafe_allow_html=True)

if len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div class="hero-container" translate="no">
        <div class="hero-title">{t["hero_title"]}</div>
        <div class="hero-subtitle">{t["hero_subtitle"]}</div>
    </div>
    """, unsafe_allow_html=True)

if prompt := st.chat_input(t["chat_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()
    
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant", avatar="🌍"):
        with st.spinner(t["analyzing"]):
            router_res = get_llm_intent_and_location(prompt, st.session_state.messages[:-1])
            intent = router_res.get("intent", "general_chat")
            parsed_location = router_res.get("location", None)
            filters = router_res.get("filters", {})

            if parsed_location:
                st.session_state.current_location = parsed_location
            
            effective_location = st.session_state.current_location

            if intent == "exit":
                response_text = t["exit_msg"]
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
            
            elif intent == "unsupported_location":
                response_text = t["unsupported_loc"]
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
                
            elif intent == "missing_location" and not effective_location:
                response_text = t["missing_loc"]
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
                
            elif intent in ["general_chat", "score_explanation"]:
                answer_container = st.empty()
                generator = generate_conversational_answer(prompt, t["code"], st.session_state.messages[:-1])
                full_answer = ""
                think_text = ""
                for chunk in generator:
                    if isinstance(chunk, dict):
                        if chunk["type"] == "think":
                            think_text += chunk["content"]
                            answer_container.markdown(f"<div style='color: #888; font-style: italic; font-size: 0.9em; margin-bottom: 10px;'>🤔 <i>Düşünüyor...</i><br>{think_text}</div>", unsafe_allow_html=True)
                        else:
                            full_answer += chunk["content"]
                            answer_container.markdown(full_answer, unsafe_allow_html=True)
                    else:
                        full_answer += chunk
                        answer_container.markdown(full_answer, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_answer})
                st.rerun()
                
            elif intent == "follow_up":
                last_hotels = None
                for msg in reversed(st.session_state.messages[:-1]):
                    if "hotels" in msg and msg["hotels"]:
                        last_hotels = msg["hotels"]
                        break
                
                if last_hotels:
                    context_parts = []
                    for i, res in enumerate(last_hotels):
                        meta = res["metadata"]
                        score_data = calculate_recommendation_score(res)
                        final_score = score_data["score"]
                        strengths = build_strengths(res, score_data)
                        cautions = build_cautions(res)
                        
                        context_str = f"Otel {i+1}: {meta.get('hotel_name')}\nLokasyon: {meta.get('location')}\nPuan: %{final_score:.1f}\n"
                        context_str += f"Öne Çıkanlar: {', '.join(strengths) if strengths else '-'}\n"
                        context_str += f"Dikkat Edilmesi Gerekenler: {', '.join(cautions) if cautions else '-'}\n"
                        context_str += f"Açıklama: {res['text']}\n"
                        context_parts.append(context_str)
                    
                    full_context = "\n\n".join(context_parts)
                    answer_container = st.empty()
                    generator = generate_followup_answer(prompt, full_context, t["code"], st.session_state.messages[:-1])
                    full_answer = ""
                    think_text = ""
                    for chunk in generator:
                        if isinstance(chunk, dict):
                            if chunk["type"] == "think":
                                think_text += chunk["content"]
                                answer_container.markdown(f"<div style='color: #888; font-style: italic; font-size: 0.9em; margin-bottom: 10px;'>🤔 Düşünüyor...<br>{think_text}</div>", unsafe_allow_html=True)
                            else:
                                full_answer += chunk["content"]
                                answer_container.markdown(full_answer, unsafe_allow_html=True)
                        else:
                            full_answer += chunk
                            answer_container.markdown(full_answer, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "hotels": last_hotels
                    })
                else:
                    answer_container = st.empty()
                    generator = generate_conversational_answer(prompt, t["code"], st.session_state.messages[:-1])
                    full_answer = ""
                    think_text = ""
                    for chunk in generator:
                        if isinstance(chunk, dict):
                            if chunk["type"] == "think":
                                think_text += chunk["content"]
                                answer_container.markdown(f"<div style='color: #888; font-style: italic; font-size: 0.9em; margin-bottom: 10px;'>🤔 Düşünüyor...<br>{think_text}</div>", unsafe_allow_html=True)
                            else:
                                full_answer += chunk["content"]
                                answer_container.markdown(full_answer, unsafe_allow_html=True)
                        else:
                            full_answer += chunk
                            answer_container.markdown(full_answer, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full_answer})
                    st.rerun()
                
            else:
                results = search(prompt, location_filter=effective_location, filters=filters, top_k_hotels=4)
                
                if not results:
                    response_text = t["not_found"].format(effective_location or '')
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
                else:
                    context_parts = []
                    for i, res in enumerate(results):
                        meta = res["metadata"]
                        score_data = calculate_recommendation_score(res)
                        final_score = score_data["score"]
                        
                        st.markdown("---")
                        st.markdown(f"#### 🏨 {meta.get('hotel_name', t['unknown_hotel'])}")
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.write(f"📍 **{meta.get('location', '')}** | {t['class_label']}: {meta.get('hotel_class', '')}")
                            st.write(f"🛠️ **{t['amenities_label']}:** {', '.join(meta.get('amenities', [])[:6])}")
                        with c2:
                            st.metric(t["score_label"], f"%{final_score:.1f}")
                            booking_url = f"https://www.booking.com/searchresults.html?ss={meta.get('hotel_name', '').replace(' ', '+')}"
                            st.markdown(f"👉 **[{t['book_now']}]({booking_url})**", unsafe_allow_html=True)
                            
                        strengths = build_strengths(res, score_data)
                        cautions = build_cautions(res)
                        
                        context_str = f"Otel {i+1}: {meta.get('hotel_name')}\n"
                        context_str += f"Lokasyon: {meta.get('location')}\n"
                        context_str += f"Puan: %{final_score:.1f}\n"
                        context_str += f"Öne Çıkanlar: {', '.join(strengths) if strengths else '-'}\n"
                        context_str += f"Dikkat Edilmesi Gerekenler: {', '.join(cautions) if cautions else '-'}\n"
                        context_str += f"Açıklama: {res['text']}\n"
                        context_parts.append(context_str)

                    full_context = "\n\n".join(context_parts)
                    st.markdown("---")
                    
                    answer_container = st.empty()
                    generator = generate_llm_answer(prompt, full_context, st.session_state.messages[:-1], effective_location, t["code"])
                    full_answer = ""
                    think_text = ""
                    for chunk in generator:
                        if isinstance(chunk, dict):
                            if chunk["type"] == "think":
                                think_text += chunk["content"]
                                answer_container.markdown(f"<div style='color: #888; font-style: italic; font-size: 0.9em; margin-bottom: 10px;'>🤔 Düşünüyor...<br>{think_text}</div>", unsafe_allow_html=True)
                            else:
                                full_answer += chunk["content"]
                                answer_container.markdown(full_answer, unsafe_allow_html=True)
                        else:
                            full_answer += chunk
                            answer_container.markdown(full_answer, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "hotels": results
                    })
                    st.rerun()


st.markdown(f"<div class='custom-footer'>{t['footer']}</div>", unsafe_allow_html=True)
