import random
import json

cities_tr = ["New York City", "Chicago", "San Francisco", "Boston", "Washington DC", "San Diego", "Dallas", "Houston", "Denver", "Los Angeles", "Seattle", "San Antonio", "Phoenix", "Philadelphia", "Memphis", "Baltimore", "San Jose", "Detroit", "Austin", "Indianapolis", "Jacksonville", "Charlotte", "Columbus", "Fort Worth", "El Paso"]

logical_combos = [
    ("lüks", ["havuz", "spa", "oda servisi", "bar", "restoran"]),
    ("ucuz", ["ücretsiz otopark", "kahvaltı", "wifi"]),
    ("iş seyahati için", ["iş merkezi", "spor salonu", "wifi"]),
    ("aile için", ["havuz", "mutfak", "ücretsiz otopark", "evcil hayvan dostu"])
]

def generate_question(language, is_out_of_scope=False, is_unsupported=False):
    if is_out_of_scope:
        if language == "tr":
            return random.choice([
                "Bana ıslak kek tarifi verir misin?",
                "New York'a uçak biletleri ne kadar?",
                "Araba kiralama fiyatları nedir?",
                "En iyi İtalyan restoranı nerede?"
            ])
        else:
            return random.choice([
                "Can you give me a brownie recipe?",
                "How much are flights to New York?",
                "What are the car rental prices?",
                "Where is the best Italian restaurant?"
            ])
            
    if is_unsupported:
        city = random.choice(["Miami", "Paris", "Londra", "İstanbul"])
        if language == "tr":
            return f"{city}'de deniz kenarında otel arıyorum."
        else:
            return f"I'm looking for a hotel by the sea in {city}."
            
    city = random.choice(cities_tr)
    style, amenities = random.choice(logical_combos)
    
    chosen_amenities = random.sample(amenities, k=random.choice([1, 2]))
    
    if language == "tr":
        if len(chosen_amenities) == 1:
            return f"{city}'da {style} ve {chosen_amenities[0]} olan bir otel arıyorum."
        else:
            return f"{city}'da {style}, {chosen_amenities[0]} ve {chosen_amenities[1]} olan oteller nelerdir?"
    else:
        style_en = {"lüks": "luxury", "ucuz": "cheap", "iş seyahati için": "business", "aile için": "family-friendly"}[style]
        am_en_map = {
            "havuz": "pool", "spa": "spa", "oda servisi": "room service", "bar": "bar", "restoran": "restaurant",
            "ücretsiz otopark": "free parking", "kahvaltı": "breakfast", "wifi": "wifi",
            "iş merkezi": "business center", "spor salonu": "gym", "mutfak": "kitchen", "evcil hayvan dostu": "pet friendly"
        }
        en_am1 = am_en_map[chosen_amenities[0]]
        if len(chosen_amenities) == 1:
            return f"I'm looking for a {style_en} hotel in {city} with a {en_am1}."
        else:
            en_am2 = am_en_map[chosen_amenities[1]]
            return f"What are some {style_en} hotels in {city} that have a {en_am1} and {en_am2}?"

def generate_questions(lang="tr", num_questions=240):
    questions = set()
    while len(questions) < num_questions:
        r = random.random()
        if r < 0.1:
            q = generate_question(lang, is_out_of_scope=True)
        elif r < 0.2:
            q = generate_question(lang, is_unsupported=True)
        else:
            q = generate_question(lang)
            
        questions.add(q)
    return list(questions)

if __name__ == "__main__":
    suites = {}
    
    # 5 Turkish suites (150 questions)
    tr_questions = generate_questions("tr", 150)
    for i in range(5):
        suites[f"test_{i+1}_tr"] = tr_questions[i*30:(i+1)*30]
        
    # 2 English suites (50 questions - 25 each)
    en_questions = generate_questions("en", 50)
    for i in range(2):
        suites[f"test_{i+6}_en"] = en_questions[i*25:(i+1)*25]
        
    with open("tests/test_suite.json", "w", encoding="utf-8") as f:
        json.dump(suites, f, ensure_ascii=False, indent=4)
        
    print("200 logical questions generated and saved to tests/test_suite.json")
