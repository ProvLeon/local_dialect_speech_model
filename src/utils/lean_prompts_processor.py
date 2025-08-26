#!/usr/bin/env python3
"""
lean_prompts_processor.py

Processor for the new lean e-commerce Twi prompt specification (prompts_lean.csv).

CSV Schema (headers):
    id,text,canonical_intent,slot_type,slot_value,notes

Differences vs legacy processor:
    - Uses 'canonical_intent' instead of 'intent'
    - Adds optional slot columns: slot_type, slot_value
    - No 'section_id' or 'meaning' fields
    - Focuses on a lean, production-oriented intent taxonomy
    - Slot extraction is handled as metadata for downstream post-processing

Outputs (in output_dir):
    intent_mapping.json          -> { intent: index }
    label_map.json / .npy        -> Same as intent_mapping for model compatibility
    training_metadata.json       -> [{ id, text, intent, slot_type, slot_value }]
    text_to_intent.json          -> { text: canonical_intent }
    slot_statistics.json         -> Summary counts of slots
    statistics.json              -> General dataset statistics
    processed_prompts.csv        -> Normalized copy with enforced columns

Usage (programmatic):
    from src.utils.lean_prompts_processor import LeanPromptsProcessor
    p = LeanPromptsProcessor(csv_path="prompts_lean.csv")
    saved_files, stats = p.process()

CLI:
    python -m src.utils.lean_prompts_processor --csv prompts_lean.csv --output data/lean_processed_prompts
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
# NOTE: Added for static analyzers clarity (no functional change)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------
@dataclass
class LeanPromptRecord:
    """Normalized representation of a single lean prompt row."""
    id: str
    text: str
    intent: str
    slot_type: Optional[str] = None
    slot_value: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------
# Processor Class
# ---------------------------------------------------------------------
class LeanPromptsProcessor:
    """
    Processor for lean prompt CSVs with slot annotations.
    Produces training metadata and label maps for downstream model training.
    """

    REQUIRED_COLUMNS = ["id", "text", "canonical_intent"]
    OPTIONAL_COLUMNS = ["slot_type", "slot_value", "notes"]

    def __init__(self,
                 csv_path: str = "prompts_lean.csv",
                 output_dir: str = "data/lean_processed_prompts",
                 min_samples_per_intent: int = 1,
                 merge_rare: bool = False,
                 rare_intent_fallback: str = "_other_"):
        """
        Initialize lean prompts processor.

        Args:
            csv_path: Path to lean prompts CSV file.
            output_dir: Output directory for processed artifacts.
            min_samples_per_intent: Minimum samples required per intent (for stats / warnings).
            merge_rare: If True, intents below min_samples_per_intent will be merged into fallback.
            rare_intent_fallback: Label used if merge_rare=True and an intent is too rare.
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.min_samples_per_intent = min_samples_per_intent
        self.merge_rare = merge_rare
        self.rare_intent_fallback = rare_intent_fallback

        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize as empty DataFrame instead of Optional to satisfy strict typing on later reassignment
        self.df: pd.DataFrame = pd.DataFrame()
        self.intent_mapping: Dict[str, int] = {}
        self.text_to_intent: Dict[str, str] = {}
        self.records: List[LeanPromptRecord] = []
        self.intent_counts: Counter = Counter()
        self.slot_type_counts: Counter = Counter()
        self.slot_value_counts: Dict[str, Counter] = defaultdict(Counter)  # slot_type -> Counter(values)

        self._load_csv()

    # -----------------------------------------------------------------
    # Loading & Validation
    # -----------------------------------------------------------------
    def _load_csv(self):
        """Load the CSV with comment-line support and validate schema."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Lean prompts CSV not found: {self.csv_path}")

        try:
            self.df = pd.read_csv(self.csv_path, comment="#", dtype=str).fillna("")
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV '{self.csv_path}': {e}") from e

        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found columns: {list(self.df.columns)}")

        # Normalize column names (strip whitespace)
        self.df.columns = [c.strip() for c in self.df.columns]

        # Basic cleaning
        self.df["text"] = self.df["text"].astype(str).str.strip()
        self.df["canonical_intent"] = self.df["canonical_intent"].astype(str).str.strip()
        self.df["id"] = self.df["id"].astype(str).str.strip()

        # Standardize empty strings to ""
        for col in self.OPTIONAL_COLUMNS:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()

        # Filter out rows without text or intent
        before = len(self.df)
        # Filter and copy to keep type as DataFrame (avoid potential typed Series confusion)
        self.df = self.df.loc[(self.df["text"] != "") & (self.df["canonical_intent"] != "")].copy()
        after = len(self.df)
        if after < before:
            logger.info(f"Filtered {before - after} rows missing text or canonical_intent")

        logger.info(f"Loaded lean prompts CSV: {after} usable rows")

    # -----------------------------------------------------------------
    # Processing Steps
    # -----------------------------------------------------------------
    def _build_records(self):
        """Convert dataframe rows into LeanPromptRecord entries."""
        self.records = []
        for _, row in self.df.iterrows():
            rec = LeanPromptRecord(
                id=row["id"],
                text=row["text"],
                intent=row["canonical_intent"],
                slot_type=row.get("slot_type", "") or None,
                slot_value=row.get("slot_value", "") or None,
                notes=row.get("notes", "") or None
            )
            self.records.append(rec)

        logger.info(f"Built {len(self.records)} normalized prompt records")

    def _analyze_distribution(self):
        """Compute distributions for intents and slots."""
        self.intent_counts = Counter(r.intent for r in self.records)
        for intent, count in self.intent_counts.items():
            if count < self.min_samples_per_intent:
                logger.warning(f"Intent '{intent}' has only {count} samples (< {self.min_samples_per_intent})")

        for r in self.records:
            if r.slot_type:
                self.slot_type_counts[r.slot_type] += 1
                if r.slot_value:
                    self.slot_value_counts[r.slot_type][r.slot_value] += 1

        if self.merge_rare:
            self._merge_rare_intents()

    def _merge_rare_intents(self):
        """Optionally merge rare intents into fallback."""
        rare_intents = {i for i, c in self.intent_counts.items() if c < self.min_samples_per_intent}
        if not rare_intents:
            return

        logger.info(f"Merging {len(rare_intents)} rare intents into '{self.rare_intent_fallback}'")
        for r in self.records:
            if r.intent in rare_intents:
                r.intent = self.rare_intent_fallback

        # Recompute counts after merge
        self.intent_counts = Counter(r.intent for r in self.records)

    def _create_intent_mapping(self):
        """Create numeric mapping for canonical intents."""
        intents_sorted = sorted(self.intent_counts.keys())
        self.intent_mapping = {intent: idx for idx, intent in enumerate(intents_sorted)}
        logger.info(f"Intent mapping created with {len(self.intent_mapping)} intents")

    def _create_text_to_intent(self):
        """Map exact text to intent (last occurrence wins if duplicates)."""
        self.text_to_intent = {}
        for r in self.records:
            self.text_to_intent[r.text] = r.intent
        logger.info(f"text_to_intent mapping size: {len(self.text_to_intent)}")

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------
    def _save_processed_csv(self) -> str:
        """Save normalized version of prompts for reproducibility."""
        out_path = os.path.join(self.output_dir, "processed_prompts.csv")
        rows = []
        for r in self.records:
            rows.append({
                "id": r.id,
                "text": r.text,
                "intent": r.intent,
                "slot_type": r.slot_type or "",
                "slot_value": r.slot_value or "",
                "notes": r.notes or ""
            })
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def _save_json(self, name: str, data) -> str:
        path = os.path.join(self.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _save_label_map(self) -> Tuple[str, str]:
        """Save label map as JSON and NPY for compatibility.
        Store dict inside 1-element object array to satisfy strict static analyzers expecting ndarray-like."""
        json_path = self._save_json("label_map.json", self.intent_mapping)
        npy_path = os.path.join(self.output_dir, "label_map.npy")
        arr = np.array([self.intent_mapping], dtype=object)
        np.save(npy_path, arr, allow_pickle=True)
        return json_path, npy_path

    def _save_training_metadata(self) -> str:
        """Save training metadata (list of records) with slot info included."""
        metadata = []
        for r in self.records:
            metadata.append({
                "id": r.id,
                "text": r.text,
                "intent": r.intent,
                "slot_type": r.slot_type,
                "slot_value": r.slot_value,
                "notes": r.notes
            })
        return self._save_json("training_metadata.json", metadata)

    def _save_slot_statistics(self) -> str:
        """Save detailed slot usage statistics."""
        slot_stats = {
            "slot_type_counts": dict(self.slot_type_counts),
            "slot_value_counts": {k: dict(v) for k, v in self.slot_value_counts.items()},
            "total_slot_annotated": int(sum(self.slot_type_counts.values()))
        }
        return self._save_json("slot_statistics.json", slot_stats)

    def _save_general_statistics(self) -> str:
        """Save overall dataset statistics."""
        total = len(self.records)
        stats = {
            "total_prompts": total,
            "unique_intents": len(self.intent_counts),
            "intent_distribution": dict(self.intent_counts),
            "slot_types": list(self.slot_type_counts.keys()),
            "slot_type_counts": dict(self.slot_type_counts),
            "min_samples_per_intent_threshold": self.min_samples_per_intent,
            "num_with_slots": int(sum(self.slot_type_counts.values())),
            "proportion_with_slots": round(
                (sum(self.slot_type_counts.values()) / total) if total else 0.0, 4
            )
        }
        return self._save_json("statistics.json", stats)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def process(self) -> Tuple[Dict[str, str], Dict]:
        """
        Execute full processing pipeline.

        Returns:
            saved_files: dict of saved artifact paths
            stats: dict with summarized statistics
        """
        self._build_records()
        self._analyze_distribution()
        self._create_intent_mapping()
        self._create_text_to_intent()

        # Persist artifacts
        saved_files = {}
        saved_files["processed_csv"] = self._save_processed_csv()
        saved_files["intent_mapping"] = self._save_json("intent_mapping.json", self.intent_mapping)
        label_json, label_npy = self._save_label_map()
        saved_files["label_map_json"] = label_json
        saved_files["label_map_npy"] = label_npy
        saved_files["text_to_intent"] = self._save_json("text_to_intent.json", self.text_to_intent)
        saved_files["training_metadata"] = self._save_training_metadata()
        saved_files["slot_statistics"] = self._save_slot_statistics()
        saved_files["statistics"] = self._save_general_statistics()

        # Compose stats for return
        stats = {
            "total_prompts": len(self.records),
            "unique_intents": len(self.intent_mapping),
            "intent_distribution": dict(self.intent_counts),
            "slot_type_counts": dict(self.slot_type_counts),
            "slot_value_counts": {k: dict(v) for k, v in self.slot_value_counts.items()}
        }

        logger.info("Lean prompt processing complete:")
        logger.info(f"  Prompts: {stats['total_prompts']}")
        logger.info(f"  Intents: {stats['unique_intents']}")
        logger.info(f"  With slots: {sum(self.slot_type_counts.values())}")

        return saved_files, stats


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process lean prompts CSV with slot annotations")
    parser.add_argument("--csv", type=str, default="prompts_lean.csv", help="Path to prompts_lean.csv")
    parser.add_argument("--output", type=str, default="data/lean_processed_prompts", help="Output directory")
    parser.add_argument("--min-samples", type=int, default=1, help="Warn/merge threshold for rare intents")
    parser.add_argument("--merge-rare", action="store_true", help="Merge intents below threshold into fallback")
    parser.add_argument("--rare-fallback", type=str, default="_other_", help="Fallback intent label")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    processor = LeanPromptsProcessor(
        csv_path=args.csv,
        output_dir=args.output,
        min_samples_per_intent=args.min_samples,
        merge_rare=args.merge_rare,
        rare_intent_fallback=args.rare_fallback
    )
    saved, stats = processor.process()

    print("\n" + "=" * 60)
    print("LEAN PROMPTS PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Unique intents: {stats['unique_intents']}")
    print(f"Top intents:")
    for intent, count in sorted(stats['intent_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {intent}: {count}")
    print("\nSlot types:")
    for st, cnt in sorted(stats['slot_type_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {st}: {cnt}")
    print("\nArtifacts saved:")
    for k, v in saved.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
