#!/usr/bin/env python3
"""
slot_extractor.py

Utility for post-intent slot parsing for the lean Twi e-commerce voice intent system.

Context:
  The speech model first classifies an utterance to a canonical intent (see prompts_lean.csv).
  Additional structured parameters (slots) such as category, brand, size, color, price_range,
  rating, sort_key, delta (increment/decrement), quantity (absolute), payment_method, qualifier,
  and context are then derived by lightweight NLP rules.

Design Goals:
  - Pure-Python, fast, no heavy ML dependencies.
  - Easy to extend: central rule registry + composable regex matchers.
  - Non-destructive: returns both raw and normalized slot values.
  - Language-aware: handles common Twi phrases and code-switching (English brand names, etc.)
  - Defensive: never throws on malformed input; returns empty dict for no matches.

Primary API:
  extract_slots(text: str, intent: str | None = None) -> SlotExtraction
    - Provides all detected slots, with optional intent-aware refinement.
  refine_with_intent(intent: str, slots: dict) -> dict
    - Applies intent-specific normalization or conflict resolution.
  normalize(text: str) -> str
    - Light canonicalization (lowercase, strip, spacing).

Usage:
  from src.utils.slot_extractor import SlotExtractor
  extractor = SlotExtractor()
  result = extractor.extract_slots("Fa boɔ ketewa to so", intent="apply_filter")
  print(result.slots)  # {'price_range': 'low'}

Returned Data Object (SlotExtraction):
  {
    'intent': <intent or None>,
    'original_text': <raw text>,
    'normalized_text': <normalized>,
    'slots': { slot_type: value | [values] },
    'debug': { 'matched_rules': [...], 'fragments': {...} }
  }

Extensibility:
  - Add or adjust patterns in SLOT_PATTERNS
  - Add enumerations in LEXICONS
  - Add transformation logic in _post_process_slots or intent refinement mapping

Limitations:
  - Rule-based, so ambiguous multi-slot phrases may produce partial extraction.
  - Does not perform fuzzy spelling distance; relies on curated patterns.

"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Callable, Any, Tuple, Union


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SlotExtraction:
    intent: Optional[str]
    original_text: str
    normalized_text: str
    slots: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def get(self, slot_type: str, default=None):
        return self.slots.get(slot_type, default)


# ---------------------------------------------------------------------------
# Lexicons / Normalization Dictionaries
# ---------------------------------------------------------------------------

# Twi number word mapping (add more as needed)
TWI_NUMBERS = {
    "baako": 1,
    "baako pɛ": 1,
    "baako bio": 1,
    "mmienu": 2,
    "mminu": 2,
    "mmiɛnsa": 3,
    "mianu": 5,  # possible mishearing (placeholder)
    "enum": 5,
    "anan": 4,
    "nan": 4,
    "asia": 6,
    "nsia": 6,
    "ason": 7,
    "awotwe": 8,
    "nkron": 9,
    "du": 10,
}

# Category synonyms (Twi -> canonical)
CATEGORY_SYNONYMS = {
    r"\bntade[eɛ]*\b": "clothing",
    r"\bmfidie\b": "electronics",
    r"\bmpaboa\b": "shoes",
    r"\bfie\s+nnoɔma\b": "home_items",
    r"\bnnuane\b": "groceries",
    r"\bakyɛde[eɛ]*\b": "gifts",
}

BRANDS = [
    "apple", "nike", "adidas", "samsung", "tecno", "infinix"
]

COLORS = {
    r"\btuntum\b": "black",
    r"\bfitaa\b": "white",
    r"\b(asikre|kɔkɔɔ)\b": "red_orange",  # placeholder grouping
    r"\banyɛ\s*den\b": "light",
}

SIZES = {
    r"\bkɛse\b": "large",
    r"\bketewa\b": "small",
    r"\bmedium\b": "medium",
    r"\bextra\s+large\b": "xlarge",
    r"\bextra\s+small\b": "xsmall",
}

QUALIFIERS = {
    r"\bfofor[ɔɔ]\b": "new",
    r"\bcheap\b": "cheap",
    r"\bflash\s+sale\b": "flash_sale",
    r"\bdeals?\b": "deals",
    r"\bdiscounts?\b": "discounts",
}

PRICE_RANGE = {
    r"\bboɔ\s+ketewa\b": "low",
    r"\bboɔ\s+kɛse\b": "high",
}

RATING = {
    r"\brating\s+kɛse\b": "high",
    r"\brating\s+fakye\b": "low",  # example placeholder
}

SORT_KEYS = {
    r"\bprice\s+ketewa\b": "price_asc",
    r"\bprice\s+kɛse\b": "price_desc",
    r"\bfofor[ɔɔ]\b": "newest",
    r"\bpopular\b": "popular",
    r"\brating\s+kɛse\b": "rating_desc",
}

PAYMENT_METHODS = {
    r"\bmobile\s+money\b": "mobile_money",
    r"\bmomo\b": "mobile_money",
    r"\bcard\b": "card",
}

CONTEXT_KEYS = {
    r"\breceipt\b": "receipt",
    r"\border\s+details?\b": "order_details",
    r"\bpage\b": "page",
}

# For absolute quantity "Hyɛ dodow no yɛ mmiɛnsa" or "Hyɛ dodow no yɛ enum"
ABS_QUANTITY_PATTERN = re.compile(r"\bhyɛ\s+dodow\s+no\s+y[ɛe]\s+([a-zɔ]+(?:\s+[a-zɔ]+)?)", re.IGNORECASE)

# Delta patterns for change_quantity (increment/decrement)
DELTA_INCREASE_PATTERNS = [
    r"\bfa\s+baako\s+ka\s+ho\b",
    r"\bfa\s+baako\s+bio\b",
    r"\bfa\s+mmienu\s+bio\b",
    r"\bfa\s+mmiɛnsa\s+bio\b",
    r"\bka\s+ho\b",
]
DELTA_DECREASE_PATTERNS = [
    r"\byi\s+baako(\s+p[ɛe])?\b",
    r"\byi\s+mmienu\b",
    r"\byi\s+mmiɛnsa\b",
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_number_phrase(phrase: str) -> Optional[int]:
    """
    Resolve a Twi number phrase to an integer if known.
    Attempt direct mapping first, else split tokens.
    """
    phrase_norm = phrase.strip().lower()
    if phrase_norm in TWI_NUMBERS:
        return TWI_NUMBERS[phrase_norm]

    # Multi-token fallback
    tokens = phrase_norm.split()
    total = 0
    matched = False
    for t in tokens:
        if t in TWI_NUMBERS:
            total += TWI_NUMBERS[t]
            matched = True
    return total if matched else None


# ---------------------------------------------------------------------------
# Slot Pattern Framework
# ---------------------------------------------------------------------------

class SlotRule:
    """
    Encapsulates a single slot extraction rule.

    Attributes:
        slot_type: The slot category (e.g. 'color')
        pattern: Compiled regex
        value: Optional static value override. If None, uses captured group or canonical mapping.
        func: Optional custom extraction function
        multi: Whether multiple matches can be accumulated
    """
    __slots__ = ("slot_type", "pattern", "value", "func", "multi")

    def __init__(self,
                 slot_type: str,
                 pattern: Union[str, Pattern],
                 value: Optional[str] = None,
                 func: Optional[Callable[[re.Match, Dict[str, Any]], Any]] = None,
                 multi: bool = False):
        self.slot_type = slot_type
        self.pattern = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
        self.value = value
        self.func = func
        self.multi = multi

    def apply(self, text: str, slots: Dict[str, Any], debug_matches: List[str]):
        for match in self.pattern.finditer(text):
            if self.func:
                extracted = self.func(match, slots)
            elif self.value is not None:
                extracted = self.value
            else:
                # Default: use matched group or whole match
                extracted = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if extracted is None:
                continue
            if self.multi:
                slots.setdefault(self.slot_type, [])
                if extracted not in slots[self.slot_type]:
                    slots[self.slot_type].append(extracted)
            else:
                # Do not overwrite existing value unless identical
                if self.slot_type not in slots:
                    slots[self.slot_type] = extracted
            debug_matches.append(f"{self.slot_type}:{extracted} <- {match.group(0)}")


# ---------------------------------------------------------------------------
# Core Extractor
# ---------------------------------------------------------------------------

class SlotExtractor:
    """
    Main slot extraction engine.

    - Initialize once and reuse (regex precompiled).
    - Use extract_slots(text, intent=None) for primary usage.
    """

    def __init__(self):
        self.rules: List[SlotRule] = []
        self._build_static_rules()

    # --------------------------- Rule Construction ---------------------------

    def _build_static_rules(self):
        # Categories
        for pat, canon in CATEGORY_SYNONYMS.items():
            self.rules.append(SlotRule("category", pat, value=canon))

        # Brands (simple word boundary match)
        for brand in BRANDS:
            self.rules.append(SlotRule("brand", rf"\b{brand}\b", value=brand))

        # Colors
        for pat, canon in COLORS.items():
            self.rules.append(SlotRule("color", pat, value=canon))

        # Sizes
        for pat, canon in SIZES.items():
            self.rules.append(SlotRule("size", pat, value=canon))

        # Qualifiers
        for pat, canon in QUALIFIERS.items():
            self.rules.append(SlotRule("qualifier", pat, value=canon, multi=True))

        # Price range
        for pat, canon in PRICE_RANGE.items():
            self.rules.append(SlotRule("price_range", pat, value=canon))

        # Rating
        for pat, canon in RATING.items():
            self.rules.append(SlotRule("rating", pat, value=canon))

        # Sort keys
        for pat, canon in SORT_KEYS.items():
            self.rules.append(SlotRule("sort_key", pat, value=canon))

        # Payment methods
        for pat, canon in PAYMENT_METHODS.items():
            self.rules.append(SlotRule("payment_method", pat, value=canon))

        # Context keys
        for pat, canon in CONTEXT_KEYS.items():
            self.rules.append(SlotRule("context", pat, value=canon))

        # Delta (increment)
        for pat in DELTA_INCREASE_PATTERNS:
            self.rules.append(SlotRule("delta", pat, func=lambda m, _: +1, multi=True))

        # Delta (decrement)
        for pat in DELTA_DECREASE_PATTERNS:
            self.rules.append(SlotRule("delta", pat, func=lambda m, _: -1, multi=True))

        # Absolute quantity (custom function)
        self.rules.append(SlotRule(
            "quantity",
            ABS_QUANTITY_PATTERN,
            func=self._extract_abs_quantity
        ))

    # ---------------------------- Custom Functions ---------------------------

    def _extract_abs_quantity(self, match: re.Match, slots: Dict[str, Any]):
        phrase = match.group(1)
        num = resolve_number_phrase(phrase)
        return num

    # --------------------------- Public Interface ---------------------------

    def extract_slots(self, text: str, intent: Optional[str] = None) -> SlotExtraction:
        original_text = text
        norm = normalize_text(text)
        slots: Dict[str, Any] = {}
        debug_matches: List[str] = []

        # Apply pattern rules
        for rule in self.rules:
            rule.apply(norm, slots, debug_matches)

        # Post-process derived values
        self._post_process_slots(slots)

        # Intent-aware refinement
        refined = self.refine_with_intent(intent, slots)

        return SlotExtraction(
            intent=intent,
            original_text=original_text,
            normalized_text=norm,
            slots=refined,
            debug={
                "matched_rules": debug_matches,
                "intermediate_slots": slots
            }
        )

    def refine_with_intent(self, intent: Optional[str], slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply intent-specific adjustments.
        Example: For change_quantity, convert multiple deltas to a sum.
        """
        if not intent:
            return slots

        refined = dict(slots)

        if intent == "change_quantity" and "delta" in refined:
            deltas = refined["delta"]
            if isinstance(deltas, list):
                refined["delta"] = sum(deltas)
            # Remove absolute quantity if both present
            if "quantity" in refined:
                # Keep absolute only if no delta? Business decision: prefer delta
                refined.pop("quantity", None)

        if intent == "set_quantity" and "quantity" not in refined:
            # If user said "Hyɛ dodow no yɛ mmiɛnsa" it would be captured; else try fallback parse
            q = self._fallback_quantity_parse(refined.get("_raw_text", ""))
            if q:
                refined["quantity"] = q

        # Search intents: unify qualifier multi-values
        if intent in ("search", "apply_filter") and "qualifier" in refined:
            vals = refined["qualifier"]
            if isinstance(vals, list) and len(vals) == 1:
                refined["qualifier"] = vals[0]

        return refined

    def normalize(self, text: str) -> str:
        return normalize_text(text)

    # ----------------------------- Post Processing ---------------------------

    def _post_process_slots(self, slots: Dict[str, Any]):
        # If delta list empty after logic, remove
        if "delta" in slots and not slots["delta"]:
            slots.pop("delta", None)

        # Deduplicate multi lists
        for k, v in list(slots.items()):
            if isinstance(v, list):
                slots[k] = list(dict.fromkeys(v))  # preserve order

        # Remove None values
        for k in list(slots.keys()):
            if slots[k] is None:
                slots.pop(k, None)

    # ----------------------------- Fallback Parsers -------------------------

    def _fallback_quantity_parse(self, text: str) -> Optional[int]:
        # Attempt naive parse if needed
        for word, val in TWI_NUMBERS.items():
            if re.search(rf"\b{re.escape(word)}\b", text.lower()):
                return val
        return None


# ---------------------------------------------------------------------------
# Convenience Singleton
# ---------------------------------------------------------------------------

_default_extractor: Optional[SlotExtractor] = None

def get_slot_extractor() -> SlotExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = SlotExtractor()
    return _default_extractor


# ---------------------------------------------------------------------------
# CLI / Demo
# ---------------------------------------------------------------------------

def _demo():
    extractor = get_slot_extractor()
    samples = [
        ("Fa boɔ ketewa to so", "apply_filter"),
        ("Sort by price kɛse mu", "sort_items"),
        ("Fa mmiɛnsa bio ka ho", "change_quantity"),
        ("Hyɛ dodow no yɛ enum", "set_quantity"),
        ("Fa color tuntum", "select_color"),
        ("Hwehwɛ mfidie", "search"),
        ("Fa MoMo", "make_payment"),
        ("Kyerɛ me receipt no", "show_description"),
        ("Fa rating kɛse di kan", "sort_items"),
        ("Fa size ketewa", "select_size"),
    ]

    for text, intent in samples:
        res = extractor.extract_slots(text, intent)
        print(f"\nText: {text} | Intent: {intent}")
        print("Slots:", res.slots)
        print("Debug:", res.debug["matched_rules"])

if __name__ == "__main__":
    _demo()
