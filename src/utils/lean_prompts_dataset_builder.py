#!/usr/bin/env python
"""
lean_prompts_dataset_builder.py

Utility script to transform the lean prompt specification (prompts_lean.csv)
into artifacts needed for training the updated intent classification model:

Artifacts produced (under --out-dir, default: data/lean_dataset):
  - label_map.json            : Intent -> index mapping (sorted, stable)
  - intents.txt               : One intent per line (same order as label_map indices)
  - manifest.jsonl            : One JSON object per line describing each training sample:
                                {
                                  "id": "...",
                                  "transcript": "...",
                                  "intent": "...",
                                  "slots": {"slot_type": "...", "slot_value": "..."},
                                  "audio_path": "data/raw/<id>.wav" (if exists else None)
                                }
  - class_distribution.json   : Raw counts per intent
  - balancing_suggestions.json: Recommended target counts & oversampling factors
  - missing_audio.csv         : Rows (id,audio_path) where expected audio file not found
  - summary.txt               : Human-readable summary of all outputs

Core assumptions:
  - Each row in prompts_lean.csv has:
        id,text,canonical_intent,slot_type,slot_value,notes
  - Comments / section headers start with '#', those rows are ignored.
  - Audio files expected at: <audio_root>/<id><audio_ext> (default: data/raw/*.wav)
    unless you later create a metadata file linking actual audio recordings.

Balanced dataset strategy:
  - We DO NOT perform duplication here; we only produce recommendations.
  - You can later implement oversampling or targeted recording guided by balancing_suggestions.json.

CLI Usage:
  python -m src.utils.lean_prompts_dataset_builder \
      --prompts prompts_lean.csv \
      --audio-root data/raw \
      --out-dir data/lean_dataset \
      --audio-ext .wav \
      --min-per-class 40 \
      --label-column canonical_intent

This script uses only standard library modules for portability.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class PromptRow:
    row_id: str
    text: str
    intent: str
    slot_type: Optional[str]
    slot_value: Optional[str]
    notes: Optional[str]

    def to_manifest_record(
        self,
        audio_root: str,
        audio_ext: str
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Build JSON-serializable manifest record & indicate whether audio exists.
        """
        audio_path = os.path.join(audio_root, f"{self.row_id}{audio_ext}")
        audio_exists = os.path.isfile(audio_path)
        record = {
            "id": self.row_id,
            "transcript": self.text,
            "intent": self.intent,
            "slots": {},
            "audio_path": audio_path if audio_exists else None,
            "notes": self.notes if self.notes else ""
        }
        if self.slot_type and self.slot_type.strip():
            record["slots"]["slot_type"] = self.slot_type.strip()
        if self.slot_value and self.slot_value.strip():
            record["slots"]["slot_value"] = self.slot_value.strip()
        return record, audio_exists


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

def read_prompts_csv(csv_path: str,
                     id_col: str = "id",
                     text_col: str = "text",
                     intent_col: str = "canonical_intent",
                     slot_type_col: str = "slot_type",
                     slot_value_col: str = "slot_value",
                     notes_col: str = "notes") -> List[PromptRow]:
    """
    Read prompts CSV, ignoring commented / blank rows.
    """
    rows: List[PromptRow] = []
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Prompts file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda l: not l.strip().startswith("#"), f))
        mandatory = [id_col, text_col, intent_col]
        for m in mandatory:
            if not reader.fieldnames or m not in reader.fieldnames:
                raise ValueError(f"Required column '{m}' not present in CSV headers: {reader.fieldnames}")

        for line_idx, raw in enumerate(reader, start=2):  # account for header line
            if not raw:
                continue
            # Skip rows where id or intent is missing/empty
            rid = (raw.get(id_col) or "").strip()
            if not rid:
                continue
            intent = (raw.get(intent_col) or "").strip()
            if not intent:
                # Skip rows without an intent (maybe a header)
                continue
            text = (raw.get(text_col) or "").strip()
            slot_type = (raw.get(slot_type_col) or "").strip() or None
            slot_value = (raw.get(slot_value_col) or "").strip() or None
            notes = (raw.get(notes_col) or "").strip() or None

            rows.append(PromptRow(
                row_id=rid,
                text=text,
                intent=intent,
                slot_type=slot_type,
                slot_value=slot_value,
                notes=notes
            ))
    return rows


# --------------------------------------------------------------------------------------
# Label map
# --------------------------------------------------------------------------------------

def build_label_map(rows: List[PromptRow]) -> Dict[str, int]:
    """
    Sort intents alphabetically for stable indexing.
    """
    intents = sorted({r.intent for r in rows})
    return {intent: idx for idx, intent in enumerate(intents)}


# --------------------------------------------------------------------------------------
# Manifest generation
# --------------------------------------------------------------------------------------

def generate_manifest(rows: List[PromptRow],
                      audio_root: str,
                      audio_ext: str,
                      out_path: str) -> Dict[str, Any]:
    """
    Generate manifest.jsonl (one JSON object per line) and return stats.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    missing_audio_entries = []
    count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for r in rows:
            rec, exists = r.to_manifest_record(audio_root, audio_ext)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            if not exists:
                missing_audio_entries.append((r.row_id, rec["audio_path"]))

    return {
        "total_records": count,
        "missing_audio": missing_audio_entries
    }


# --------------------------------------------------------------------------------------
# Balancing suggestions
# --------------------------------------------------------------------------------------

def compute_class_distribution(rows: List[PromptRow]) -> Dict[str, int]:
    counter = Counter(r.intent for r in rows)
    return dict(counter)


def suggest_balancing(distribution: Dict[str, int],
                      min_per_class: int,
                      target_strategy: str = "ceil_min") -> Dict[str, Dict[str, Any]]:
    """
    Suggest oversampling / new recordings per intent.

    Strategies:
      ceil_min  : Raise all classes below min_per_class to min_per_class
      median    : Raise all below median to median
      max_match : Raise all below max count to that max (can be large)

    Returns: intent -> {current, target, deficit, oversample_factor}
    """
    if not distribution:
        return {}

    values = list(distribution.values())
    max_count = max(values)
    median_count = sorted(values)[len(values) // 2]

    suggestions: Dict[str, Dict[str, Any]] = {}

    for intent, count in distribution.items():
        if target_strategy == "max_match":
            target = max_count
        elif target_strategy == "median":
            target = max(median_count, min_per_class)
        else:  # default 'ceil_min'
            target = max(min_per_class, count)

        deficit = max(0, target - count)
        oversample_factor = round(target / count, 2) if count > 0 else None

        suggestions[intent] = {
            "current": count,
            "target": target,
            "deficit": deficit,
            "oversample_factor": oversample_factor
        }

    return suggestions


# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------

def write_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_lines(lines: List[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def write_missing_audio(missing: List[Tuple[str, str]], path: str):
    if not missing:
        write_lines(["id,audio_path"], path)
        return
    lines = ["id,audio_path"] + [f"{mid},{ap}" for mid, ap in missing]
    write_lines(lines, path)


def write_summary(out_dir: str,
                  label_map: Dict[str, int],
                  distribution: Dict[str, int],
                  balancing: Dict[str, Dict[str, Any]],
                  manifest_stats: Dict[str, Any],
                  prompts_path: str,
                  audio_root: str,
                  audio_ext: str):
    lines = []
    lines.append("Lean Prompts Dataset Build Summary")
    lines.append("=" * 40)
    lines.append(f"Prompts file        : {prompts_path}")
    lines.append(f"Output directory    : {out_dir}")
    lines.append(f"Audio root          : {audio_root}")
    lines.append(f"Audio extension     : {audio_ext}")
    lines.append("")
    lines.append(f"Intents (count)     : {len(label_map)}")
    lines.append("")
    lines.append("Intent Index Mapping:")
    for intent, idx in sorted(label_map.items(), key=lambda x: x[1]):
        lines.append(f"  {idx:02d} -> {intent}")
    lines.append("")
    lines.append("Raw Class Distribution:")
    for intent, count in sorted(distribution.items()):
        lines.append(f"  {intent}: {count}")
    lines.append("")
    lines.append("Balancing Suggestions (strategy=ceil_min):")
    for intent, info in sorted(balancing.items()):
        lines.append(f"  {intent}: current={info['current']} target={info['target']} deficit={info['deficit']} oversample_factor={info['oversample_factor']}")
    lines.append("")
    lines.append(f"Manifest records    : {manifest_stats.get('total_records', 0)}")
    lines.append(f"Missing audio files : {len(manifest_stats.get('missing_audio', []))}")
    lines.append("")
    lines.append("Next Steps:")
    lines.append("  1. Record / collect audio for missing IDs and low-count intents.")
    lines.append("  2. Re-run this builder or create incremental feature extraction pipeline.")
    lines.append("  3. Extract features (MFCC/combined) using updated feature extractor aware of canonical_intent.")
    lines.append("  4. Train model with new label_map.json and ensure slots leveraged at inference.")
    lines.append("  5. Monitor per-class F1; merge or augment classes with chronic low performance.")
    write_lines(lines, os.path.join(out_dir, "summary.txt"))


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def build_dataset(prompts_path: str,
                  out_dir: str,
                  audio_root: str,
                  audio_ext: str,
                  min_per_class: int,
                  label_column: str,
                  balancing_strategy: str = "ceil_min"):

    # 1. Read prompts
    rows = read_prompts_csv(
        prompts_path,
        id_col="id",
        text_col="text",
        intent_col=label_column,
        slot_type_col="slot_type",
        slot_value_col="slot_value",
        notes_col="notes"
    )
    if not rows:
        raise RuntimeError("No valid rows parsed from prompts CSV.")

    os.makedirs(out_dir, exist_ok=True)

    # 2. Build label map
    label_map = build_label_map(rows)
    write_json(label_map, os.path.join(out_dir, "label_map.json"))
    write_lines([intent for intent, _ in sorted(label_map.items(), key=lambda x: x[1])],
                os.path.join(out_dir, "intents.txt"))

    # 3. Manifest
    manifest_stats = generate_manifest(
        rows,
        audio_root=audio_root,
        audio_ext=audio_ext,
        out_path=os.path.join(out_dir, "manifest.jsonl")
    )

    # 4. Distribution & balancing suggestions
    distribution = compute_class_distribution(rows)
    write_json(distribution, os.path.join(out_dir, "class_distribution.json"))

    balancing = suggest_balancing(distribution, min_per_class, target_strategy=balancing_strategy)
    write_json(balancing, os.path.join(out_dir, "balancing_suggestions.json"))

    # 5. Missing audio list
    write_missing_audio(manifest_stats.get("missing_audio", []),
                        os.path.join(out_dir, "missing_audio.csv"))

    # 6. Summary
    write_summary(out_dir, label_map, distribution, balancing, manifest_stats,
                  prompts_path, audio_root, audio_ext)

    print(f"[OK] Dataset build complete. See: {out_dir}")
    print(" Key files:")
    for fname in ["label_map.json", "intents.txt", "manifest.jsonl",
                  "class_distribution.json", "balancing_suggestions.json",
                  "missing_audio.csv", "summary.txt"]:
        print(f"  - {fname}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lean prompts dataset artifacts for intent classification."
    )
    parser.add_argument("--prompts", type=str, default="prompts_lean.csv",
                        help="Path to prompts_lean.csv")
    parser.add_argument("--out-dir", type=str, default="data/lean_dataset",
                        help="Output directory for artifacts")
    parser.add_argument("--audio-root", type=str, default="data/raw",
                        help="Root directory containing audio files named <id>.wav")
    parser.add_argument("--audio-ext", type=str, default=".wav",
                        help="Audio file extension (e.g. .wav, .mp3)")
    parser.add_argument("--min-per-class", type=int, default=40,
                        help="Minimum desired samples per class for balancing suggestions")
    parser.add_argument("--label-column", type=str, default="canonical_intent",
                        help="Which column to treat as the label (default: canonical_intent)")
    parser.add_argument("--balancing-strategy", type=str, default="ceil_min",
                        choices=["ceil_min", "median", "max_match"],
                        help="Strategy for target balancing counts")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    build_dataset(
        prompts_path=args.prompts,
        out_dir=args.out_dir,
        audio_root=args.audio_root,
        audio_ext=args.audio_ext,
        min_per_class=args.min_per_class,
        label_column=args.label_column,
        balancing_strategy=args.balancing_strategy
    )


if __name__ == "__main__":
    main()
