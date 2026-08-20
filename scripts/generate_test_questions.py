import random
import json
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_suite_path = os.path.join(base_dir, 'tests', 'test_suite.json')

cities = ["New York City", "Chicago", "San Francisco", "Boston", "Washington DC", "San Diego", "Dallas", "Houston", "Denver", "Los Angeles", "Seattle", "San Antonio", "Phoenix", "Philadelphia", "Memphis", "Baltimore", "San Jose", "Detroit", "Austin", "Indianapolis", "Jacksonville", "Charlotte", "Columbus", "Fort Worth", "El Paso"]

logical_combos = [
    ("luxury", ["pool", "spa", "room service", "bar", "restaurant"]),
    ("cheap", ["free parking", "breakfast", "wifi"]),
    ("business", ["business center", "gym", "wifi"]),
    ("family-friendly", ["pool", "kitchen", "free parking", "pet friendly"])
]

def generate_question(is_out_of_scope=False, is_unsupported=False):
    if is_out_of_scope:
        return random.choice([
            "Can you give me a brownie recipe?",
            "How much are flights to New York?",
            "What are the car rental prices?",
            "Where is the best Italian restaurant?"
        ])
            
    if is_unsupported:
        city = random.choice(["Miami", "Paris", "London", "Istanbul"])
        return f"I'm looking for a hotel by the sea in {city}."
            
    city = random.choice(cities)
    style, amenities = random.choice(logical_combos)
    
    chosen_amenities = random.sample(amenities, k=random.choice([1, 2]))
    
    if len(chosen_amenities) == 1:
        return f"I'm looking for a {style} hotel in {city} with a {chosen_amenities[0]}."
    else:
        return f"What are some {style} hotels in {city} that have a {chosen_amenities[0]} and {chosen_amenities[1]}?"

def generate_questions(num_questions=200):
    questions = set()
    while len(questions) < num_questions:
        r = random.random()
        if r < 0.1:
            q = generate_question(is_out_of_scope=True)
        elif r < 0.2:
            q = generate_question(is_unsupported=True)
        else:
            q = generate_question()
            
        questions.add(q)
    return list(questions)

if __name__ == "__main__":
    suites = {}
    
    # 8 English suites (200 questions - 25 each)
    en_questions = generate_questions(200)
    for i in range(8):
        suites[f"test_{i+1}_en"] = en_questions[i*25:(i+1)*25]
        
    with open(test_suite_path, "w", encoding="utf-8") as f:
        json.dump(suites, f, indent=4)
        
    print("200 logical English questions generated and saved to tests/test_suite.json")
