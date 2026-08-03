"""Deterministic, evidence-grounded summaries for common review questions.

The local Qwen model remains available for open-ended review questions, but
common aspect questions should not depend on generation succeeding.  This
module combines the per-hotel numeric review statistics with descriptors that
are actually present in the retrieved review excerpts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATS_PATH = _PROJECT_ROOT / "data" / "hotel_review_stats.json"
_STATS_CACHE = None


_ASPECTS = (
    {
        "key": "noise",
        "question": (r"\bnoise\b", r"\bnoisy\b", r"\bquiet\b", r"\bloud\b", r"soundproof"),
        "anchors": (r"\bnoise\b", r"\bnoisy\b", r"\bquiet\b", r"\bloud\b", r"\bhear\b", r"soundproof", r"thin walls?"),
        "label": "noise levels",
        "rating_key": None,
        "rating_label": None,
        "positive": (
            ("quiet rooms", r"\b(?:very |extremely |surprisingly )?quiet\b"),
            ("little or no disruptive noise", r"\b(?:no|little) (?:outside |airport |hallway )?noise\b|\bnot noisy\b|\bnever heard (?:any )?noise\b"),
            ("good sound isolation", r"\bsoundproof(?:ed|ing)?\b|\bwell insulated\b"),
        ),
        "negative": (
            ("disruptive noise", r"\bnois(?:e|y)\b|\bloud\b"),
            ("rooms that were not quiet", r"\bnot quiet\b|\bwasn't quiet\b|\bweren't quiet\b"),
            ("thin walls", r"\bthin walls?\b"),
            ("sounds carrying into rooms", r"\bcould hear\b|\bheard (?:people|voices|traffic|planes?|doors?)\b"),
            ("sleep disturbance", r"\b(?:kept|woke) (?:me|us) (?:up|awake)\b|\bdisturb(?:ed|ing) sleep\b"),
        ),
    },
    {
        "key": "service",
        "question": (r"\bservice\b", r"\bstaff\b", r"front desk", r"concierge", r"\bvalet\b"),
        "anchors": (r"\bservice\b", r"\bstaff\b", r"front desk", r"reception", r"concierge", r"\bvalet\b", r"employees?"),
        "label": "service",
        "rating_key": "service",
        "rating_label": "service",
        "positive": (
            ("friendly staff", r"\bfriendly (?:staff|service|team|employees?)\b|\bstaff (?:was|were|are) friendly\b"),
            ("helpful staff", r"\bhelpful\b|\baccommodating\b|\bwent (?:out of|above and beyond)\b"),
            ("good service", r"\b(?:good|great|excellent|wonderful|outstanding) service\b"),
            ("attentive service", r"\battentive\b|\bcourteous\b|\bprofessional\b"),
            ("efficient service", r"\befficient\b|\bprompt service\b|\bquick service\b"),
        ),
        "negative": (
            ("poor valet service", r"\bpoor valet service\b|\bvalet[^.!?]{0,45}\b(?:poor|rude|slow|problem)\b"),
            ("rude treatment", r"\brude(?:ly)?\b|\bunfriendly\b|\bimpolite\b"),
            ("slow service", r"\bslow service\b|\bservice (?:was|is) slow\b|\blong wait\b|\bwaited[^.!?]{0,20}\b(?:minutes?|hours?)\b"),
            ("unhelpful staff", r"\bunhelpful\b|\bignored\b|\bdismissive\b"),
            ("front-desk problems", r"\bfront desk[^.!?]{0,45}\b(?:problem|issue|rude|slow|unhelpful)\b"),
        ),
    },
    {
        "key": "cleanliness",
        "question": (r"\bclean(?:liness)?\b", r"\bdirty\b", r"\bspotless\b", r"\bhygiene\b"),
        "anchors": (r"\bclean(?:liness)?\b", r"\bdirty\b", r"\bspotless\b", r"\bfilthy\b", r"\bhygiene\b"),
        "label": "cleanliness",
        "rating_key": "cleanliness",
        "rating_label": "cleanliness",
        "positive": (
            ("clean rooms", r"\bclean (?:room|rooms|property|hotel|bathroom)\b|\brooms? (?:was|were|are) clean\b"),
            ("spotless spaces", r"\bspotless\b|\bimmaculate\b"),
            ("well-maintained areas", r"\bwell[ -]maintained\b|\bwell kept\b"),
        ),
        "negative": (
            ("dirty areas", r"\bdirty\b|\bfilthy\b|\bgrimy\b"),
            ("housekeeping issues", r"\bhousekeeping[^.!?]{0,40}\b(?:issue|problem|missed|poor)\b"),
            ("stains or unpleasant odors", r"\bstains?\b|\bunpleasant (?:smell|odor)\b|\bsmelled\b"),
        ),
    },
    {
        "key": "value",
        "question": (r"\bvalue\b", r"worth (?:it|the money)", r"\bexpensive\b", r"\boverpriced\b"),
        "anchors": (r"\bvalue\b", r"worth (?:it|the money)", r"\bexpensive\b", r"\boverpriced\b", r"\bcheap\b", r"\bpricey\b"),
        "label": "value",
        "rating_key": "value",
        "rating_label": "value",
        "positive": (
            ("good value", r"\bgood value\b|\bgreat value\b|\bworth (?:it|the money)\b"),
            ("reasonable value", r"\breasonable\b|\baffordable\b"),
        ),
        "negative": (
            ("high perceived cost", r"\bexpensive\b|\bpricey\b|\boverpriced\b"),
            ("poor value", r"\bpoor value\b|\bnot worth (?:it|the money)\b"),
        ),
    },
    {
        "key": "location",
        "question": (r"\blocation\b", r"\barea\b", r"\bnearby\b", r"\bconvenien(?:t|ce)\b"),
        "anchors": (r"\blocation\b", r"\bconvenien(?:t|ce)\b", r"\bnear\b", r"\bwalk\b", r"\baccess\b", r"\bterminal\b"),
        "label": "the location",
        "rating_key": None,
        "rating_label": None,
        "positive": (
            ("a convenient location", r"\b(?:very |incredibly |extremely )?convenient\b"),
            ("easy access", r"\beasy access\b|\bgreat access\b|\bdirect access\b"),
            ("walkable connections", r"\bwalk(?:ing)? (?:distance|to)\b"),
        ),
        "negative": (
            ("an inconvenient location", r"\binconvenient\b|\bpoor location\b"),
            ("access difficulties", r"\bdifficult to (?:reach|access)\b|\bhard to (?:find|reach)\b"),
        ),
    },
    {
        "key": "breakfast",
        "question": (r"\bbreakfast\b", r"\bfood\b", r"\brestaurant\b", r"\bdining\b"),
        "anchors": (r"\bbreakfast\b", r"\bfood\b", r"\brestaurant\b", r"\bmeal\b", r"\bdining\b"),
        "label": "food and breakfast",
        "rating_key": None,
        "rating_label": None,
        "positive": (
            ("good food", r"\b(?:good|great|excellent|delicious|solid) (?:food|breakfast|meal)\b"),
            ("a well-liked restaurant", r"\b(?:good|great|excellent) restaurant\b"),
        ),
        "negative": (
            ("disappointing food", r"\b(?:bad|poor|disappointing|cold) (?:food|breakfast|meal)\b"),
            ("limited breakfast choices", r"\blimited (?:breakfast|food)\b|\bfew (?:breakfast|food) options\b"),
        ),
    },
    {
        "key": "parking",
        "question": (r"\bparking\b", r"\bvalet\b", r"park and fly"),
        "anchors": (r"\bparking\b", r"\bvalet\b", r"park and fly"),
        "label": "parking and valet service",
        "rating_key": None,
        "rating_label": None,
        "positive": (
            ("convenient parking", r"\b(?:free|easy|convenient) (?:valet )?parking\b|\bpark and fly\b"),
            ("helpful valet staff", r"\bvalet[^.!?]{0,45}\b(?:helpful|offered|great|good)\b"),
        ),
        "negative": (
            ("poor valet service", r"\bpoor valet service\b|\bvalet[^.!?]{0,45}\b(?:poor|rude|slow|problem)\b"),
            ("parking difficulties", r"\bparking[^.!?]{0,35}\b(?:difficult|problem|issue|expensive)\b"),
        ),
    },
    {
        "key": "wifi",
        "question": (r"\bwi[ -]?fi\b", r"\binternet\b", r"\bconnection\b"),
        "anchors": (r"\bwi[ -]?fi\b", r"\binternet\b", r"\bconnection\b"),
        "label": "Wi-Fi",
        "rating_key": None,
        "rating_label": None,
        "positive": (
            ("reliable Wi-Fi", r"\b(?:good|great|fast|reliable|strong) (?:wi[ -]?fi|internet|connection)\b"),
        ),
        "negative": (
            ("weak or unreliable Wi-Fi", r"\b(?:bad|poor|slow|weak|unreliable) (?:wi[ -]?fi|internet|connection)\b"),
            ("connection problems", r"\b(?:wi[ -]?fi|internet|connection)[^.!?]{0,30}\b(?:issue|problem|drop)\b"),
        ),
    },
    {
        "key": "rooms",
        "question": (r"\brooms?\b", r"\bbeds?\b", r"\bbathrooms?\b", r"\bsuites?\b"),
        "anchors": (r"\brooms?\b", r"\bbeds?\b", r"\bbathrooms?\b", r"\bsuites?\b", r"\bsleep\b"),
        "label": "the rooms",
        "rating_key": "rooms",
        "rating_label": "room",
        "positive": (
            ("clean rooms", r"\bclean (?:room|rooms|bathroom)\b|\brooms? (?:was|were|are) clean\b"),
            ("comfortable rooms or beds", r"\bcomfortable (?:room|rooms|bed|beds)\b|\b(?:room|rooms|bed|beds) (?:was|were|are) comfortable\b|\bgreat sleep\b|\bslept (?:well|great)\b"),
            ("quiet rooms", r"\b(?:room|rooms)[^.!?]{0,35}\bquiet\b|\bquiet (?:room|rooms)\b"),
            ("spacious rooms", r"\bspacious\b|\blarge (?:room|rooms)\b"),
            ("well-maintained rooms", r"\bwell[ -]maintained\b|\bmodern (?:room|rooms)\b|\bnice (?:room|rooms)\b"),
        ),
        "negative": (
            ("dark rooms", r"\bdark (?:room|rooms)\b|\brooms?[^.!?]{0,25}\bdark\b|\brooms? (?:was|were) dark\b"),
            ("small or cramped rooms", r"\b(?:small|tiny|cramped) (?:room|rooms|bathroom)\b"),
            ("dated rooms", r"\b(?:dated|outdated|worn) (?:room|rooms|furnishings)\b"),
            ("uncomfortable beds or rooms", r"\buncomfortable (?:room|rooms|bed|beds)\b"),
            ("room cleanliness problems", r"\bdirty (?:room|rooms|bathroom)\b|\brooms? (?:was|were) not clean\b"),
            ("room noise", r"\bnoisy (?:room|rooms)\b|\brooms?[^.!?]{0,30}\bnoise\b"),
        ),
    },
)


def _load_stats() -> dict:
    global _STATS_CACHE
    if _STATS_CACHE is not None:
        return _STATS_CACHE
    try:
        with _STATS_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        _STATS_CACHE = loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError, TypeError):
        _STATS_CACHE = {}
    return _STATS_CACHE


def _flatten_review_text(review_chunks) -> str:
    texts = []
    for chunk in review_chunks or []:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("chunk_text") or ""
        else:
            text = chunk
        text = str(text or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts)


def _evidence_sentences(evidence: str) -> list[str]:
    sentences = []
    seen = set()
    for part in re.split(r"(?<=[.!?])\s+|\r?\n+", evidence):
        sentence = re.sub(r"^Text:\s*", "", part.strip(), flags=re.IGNORECASE)
        if not sentence or len(sentence) < 20:
            continue
        if re.match(
            r"^(?:Review\s+\d+\s*:|Title\s*:|Source\s*:|(?:Overall|Value|Rooms?|Location|Cleanliness|Service) rating\s*:)",
            sentence,
            re.IGNORECASE,
        ):
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", sentence.casefold()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        sentences.append(sentence)
    return sentences


def _matches_any(text: str, patterns) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _has_non_negated_match(text: str, pattern: str) -> bool:
    for match in re.finditer(pattern, text, re.IGNORECASE):
        prefix = text[max(0, match.start() - 30):match.start()]
        prefix = re.split(
            r"(?:[,;]|\b(?:and|but|however)\b)", prefix, flags=re.IGNORECASE
        )[-1]
        if re.search(
            r"\b(?:no|not(?!\s+only\b)|never|hardly|isn't|wasn't|weren't|aren't|didn't)\b[^.!?]{0,22}$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        return True
    return False


def _descriptor_hits(sentences: list[str], descriptors, *, positive: bool) -> list[tuple[str, int]]:
    hits = []
    for phrase, pattern in descriptors:
        count = 0
        for sentence in sentences:
            matched = _has_non_negated_match(sentence, pattern)
            count += int(matched)
        if count:
            hits.append((phrase, count))
    return sorted(hits, key=lambda item: -item[1])


def _natural_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _rating_trend(mean: float) -> str:
    if mean >= 4.25:
        return "strongly positive"
    if mean >= 3.75:
        return "generally positive"
    if mean >= 3.25:
        return "more positive than negative"
    if mean >= 2.75:
        return "mixed"
    return "generally negative"


def _insufficient_answer(hotel_name: str, label: str) -> str:
    return (
        f"The retrieved reviews for {hotel_name} do not contain enough repeated "
        f"evidence about {label} to establish a reliable majority view. "
        "The most relevant excerpts do not provide consistent details that "
        "directly answer the question. TravelMind therefore avoids inferring "
        "anything that is not supported by those reviews."
    )


def _summarize_one_aspect(hotel_card: dict, sentences: list[str], aspect: dict) -> str:
    hotel_name = str(hotel_card.get("hotel_name") or "this hotel").strip()
    relevant = [
        sentence
        for sentence in sentences
        if _matches_any(sentence, aspect["anchors"])
    ]

    positive_hits = _descriptor_hits(
        relevant, aspect["positive"], positive=True
    )
    negative_hits = _descriptor_hits(
        relevant, aspect["negative"], positive=False
    )

    stats = _load_stats().get(str(hotel_card.get("hotel_id") or ""), {})
    rating_key = aspect["rating_key"]
    rating = stats.get(rating_key) if rating_key else None
    review_count = stats.get("review_count")

    if rating is None and not positive_hits and not negative_hits:
        return _insufficient_answer(hotel_name, aspect["label"])

    if rating is not None:
        first_sentence = (
            f"{hotel_name} has {int(review_count):,} recorded reviews, and its "
            f"available {aspect['rating_label']} ratings average "
            f"{float(rating):.2f}/5, indicating {_rating_trend(float(rating))} "
            "feedback overall."
            if review_count is not None
            else (
                f"The available {aspect['rating_label']} ratings for "
                f"{hotel_name} average {float(rating):.2f}/5, indicating "
                f"{_rating_trend(float(rating))} feedback overall."
            )
        )
    else:
        if positive_hits and negative_hits:
            balance = "mixed, with both positive and negative comments"
        elif positive_hits:
            balance = "positive in the excerpts that address it"
        elif negative_hits:
            balance = "negative in the excerpts that address it"
        else:
            balance = "not clear from the retrieved excerpts"
        first_sentence = (
            f"Among the retrieved guest excerpts for {hotel_name}, feedback "
            f"about {aspect['label']} is {balance}."
        )

    if not positive_hits and not negative_hits:
        second_sentence = (
            f"The retrieved excerpts do not contain enough topic-specific "
            f"text to explain the aggregate {aspect['label']} rating."
        )
        third_sentence = (
            "TravelMind therefore reports the aggregate rating without "
            "inferring particular strengths or complaints."
        )
    elif positive_hits:
        positive_phrases = _natural_join(
            [phrase for phrase, _ in positive_hits[:3]]
        )
        second_sentence = (
            f"Positive excerpts specifically mention {positive_phrases}."
        )
    else:
        second_sentence = (
            f"No repeated positive theme about {aspect['label']} was identified "
            "in this limited retrieved set."
        )

    if positive_hits or negative_hits:
        if negative_hits:
            negative_phrases = _natural_join(
                [phrase for phrase, _ in negative_hits[:2]]
            )
            third_sentence = (
                f"Negative excerpts also mention {negative_phrases}, so the "
                "experience is not uniformly positive."
            )
        else:
            third_sentence = (
                "No negative theme was identified in this limited retrieved "
                "set; that does not prove that no complaints exist."
            )

    return " ".join((first_sentence, second_sentence, third_sentence))


def _collapse_summary(summary: str) -> str:
    """Preserve all evidence while fitting one aspect into one sentence."""
    return summary.rstrip(".").replace(". ", "; ") + "."


def summarize_common_review_question(hotel_card, review_chunks, question: str) -> str | None:
    """Return a 3-5 sentence grounded summary for recognized aspects.

    ``None`` means the question is open-ended and should use the local LLM.
    """
    question_text = str(question or "").casefold()
    aspects = [
        candidate
        for candidate in _ASPECTS
        if _matches_any(question_text, candidate["question"])
    ]
    if "room service" in question_text:
        has_separate_room_topic = bool(
            re.search(r"\brooms\b|\bbeds?\b|\bbathrooms?\b|\bsuites?\b", question_text)
        )
        if not has_separate_room_topic:
            aspects = [aspect for aspect in aspects if aspect["key"] != "rooms"]
    if not aspects:
        return None

    hotel_card = hotel_card if isinstance(hotel_card, dict) else {}
    evidence = _flatten_review_text(review_chunks)
    sentences = _evidence_sentences(evidence)
    summaries = [
        _summarize_one_aspect(hotel_card, sentences, aspect)
        for aspect in aspects
    ]
    if len(summaries) == 1:
        return summaries[0]

    collapsed = [_collapse_summary(summary) for summary in summaries[:3]]
    scope_sentence = (
        "These statements combine aggregate ratings with themes in the limited "
        "retrieved excerpts; they do not treat top-k excerpts as a count of all "
        "textual opinions."
    )
    return " ".join(collapsed + [scope_sentence])
