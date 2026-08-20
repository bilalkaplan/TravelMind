"""
TravelMind Smoke Test Script
Runs offline validation checks without requiring Foundry/Qwen.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

passed = 0
failed = 0
total = 0

def check(label, condition):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}")

print("=" * 60)
print("TravelMind Smoke Test")
print("=" * 60)

# ── 1. Module imports ──
print("\n[1] Module Imports")
try:
    from hotel_card_builder import build_hotel_cards
    check("hotel_card_builder imports", True)
except Exception as e:
    check(f"hotel_card_builder imports ({e})", False)

try:
    from answer_validator import validate_answer
    check("answer_validator imports", True)
except Exception as e:
    check(f"answer_validator imports ({e})", False)

# ── 2. build_hotel_cards ──
print("\n[2] build_hotel_cards")
try:
    result = build_hotel_cards([])
    check("empty input returns list", isinstance(result, list) and len(result) == 0)
except Exception as e:
    check(f"empty input ({e})", False)

try:
    result = build_hotel_cards([], query_requirements={"breakfast": "REQUIRED"})
    check("accepts query_requirements kwarg", True)
except TypeError as e:
    check(f"accepts query_requirements kwarg ({e})", False)

try:
    result = build_hotel_cards([], user_query="test", requested_location="Dallas", effective_scores=None)
    check("accepts all kwargs", True)
except TypeError as e:
    check(f"accepts all kwargs ({e})", False)

# ── 4. Validator: internal analysis leak ──
print("\n[4] Validator: Internal Analysis Leak")
val = validate_answer("Okay, the user is asking for a hotel in Dallas.", [], "hotel_search", "Dallas", "en")
check("catches 'Okay, the user'", not val["passed"])

val = validate_answer("<think>I should check</think> Here is your hotel.", [], "hotel_search", "Dallas", "en")
check("catches <think> tags", not val["passed"])

val = validate_answer("Let me check the provided hotel cards for you.", [], "hotel_search", "Dallas", "en")
check("catches 'Let me check'", not val["passed"])

# ── 5. Validator: placeholder leak ──
print("\n[5] Validator: Placeholder Leak")
val = validate_answer("[Insert evidence summary here] Your hotel is ready.", [], "hotel_search", "Dallas", "en")
check("catches [Insert...", not val["passed"])

val = validate_answer("TravelMind is analyzing your request still.", [], "hotel_search", "Dallas", "en")
check("catches 'TravelMind is analyzing'", not val["passed"])

# ── 6. Validator: score overflow ──
print("\n[6] Validator: Score Overflow")
val = validate_answer("This hotel has a score of 193/100.", [], "hotel_search", "Dallas", "en")
check("catches 193/100", not val["passed"])

val = validate_answer("TravelMind Score: 91.9/100", [], "hotel_search", "Dallas", "en")
check("allows 91.9/100", val["passed"])

# ── 7. Validator: price/booking ──
print("\n[7] Validator: Price & Booking")
val = validate_answer("Book Now at booking.com for $120 per night.", [], "hotel_search", "Dallas", "en")
check("catches Book Now + booking.com", not val["passed"])

val = validate_answer("TravelMind does not provide live prices.", [], "hotel_search", "Dallas", "en")
check("allows price refusal", val["passed"])

# ── 8. Validator: room guarantee ──
print("\n[8] Validator: Room Guarantee")
cards_unknown_room = [{"hotel_name": "Test Hotel", "room_info": {"single_room": "UNKNOWN"}, "amenities": {}}]
val = validate_answer("The single room is available at Test Hotel.", cards_unknown_room, "hotel_search", "Dallas", "en")
check("catches false single room claim (UNKNOWN)", not val["passed"])

# ── 9. Validator: breakfast hallucination ──
print("\n[9] Validator: Breakfast Hallucination")
cards_no_breakfast = [{"hotel_name": "Test Hotel", "amenities": {"breakfast": "UNKNOWN"}, "room_info": {}}]
val = validate_answer("This hotel has breakfast included.", cards_no_breakfast, "hotel_search", "Dallas", "en")
check("catches breakfast claim when UNKNOWN", not val["passed"])

cards_with_breakfast = [{"hotel_name": "Test Hotel", "amenities": {"breakfast": "YES"}, "room_info": {}}]
val = validate_answer("This hotel has breakfast included.", cards_with_breakfast, "hotel_search", "Dallas", "en")
check("allows breakfast claim when YES", val["passed"])

# ── 10. Validator: map link hallucination ──
print("\n[10] Validator: Map Link Hallucination")
cards_no_map = [{"hotel_name": "Test Hotel", "map_link_type": "UNKNOWN", "amenities": {}, "room_info": {}}]
val = validate_answer("View on google.com/maps/test", cards_no_map, "hotel_search", "Dallas", "en")
check("catches map link when all UNKNOWN", not val["passed"])

# ── 11. Validator: clean answer passes ──
print("\n[11] Validator: Clean Answer")
good_cards = [{"hotel_name": "Embassy Suites Hotel", "travelmind_score": 91.9, "amenities": {"breakfast": "YES"}, "room_info": {"single_room": "UNKNOWN"}, "map_link_type": "search"}]
val = validate_answer(
    "Based on TravelMind data, Embassy Suites Hotel is a strong match with a score of 91.9/100. "
    "Breakfast is confirmed in the amenities data. Single room availability cannot be confirmed from the current dataset.",
    good_cards, "hotel_search", "Dallas", "en"
)
check("clean answer passes validation", val["passed"])

# ── 13. Fast Router ──
print("\n[13] Fast Router")
from cmu_rag_answer import fast_route_query
check("fast route price", fast_route_query("What is the price?")["intent"] == "price_question")
check("fast route pool", fast_route_query("Does it have a pool?")["intent"] == "followup_pool")
check("fast route next/other hotel", fast_route_query("Show me the other hotel")["intent"] == "follow_up")
check("fast route unsupported location", fast_route_query("Suggest a hotel in Paris")["intent"] == "unsupported_location")
check("fast route fell through", fast_route_query("Suggest a hotel") is None)

# ── 14. Prompt Builders Safety ──
print("\n[14] Prompt Builders Safety")
from prompt_builders import build_core_system_prompt
sys_prompt = build_core_system_prompt().lower()
check("prompt_builders has no <think> tag", "<think>" not in sys_prompt)
check("prompt_builders has no 'wrap your reasoning'", "wrap your reasoning" not in sys_prompt)
check("prompt_builders has no 'temporary filtering event'", "temporary filtering event" not in sys_prompt)

# ── Summary ──
print("\n" + "=" * 60)
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {failed} test(s) failed")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
