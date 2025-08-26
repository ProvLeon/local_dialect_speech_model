#!/usr/bin/env python3
"""
build_audio_manifest.py

Build a unified audio manifest for the lean prompts dataset.

Why this exists:
  - Your lean prompts file (prompts_lean.csv) specifies prompt IDs and intents
    but not the actual recorded audio file paths.
  - The recording pipeline saves canonical per-participant copies at:
        data/raw/<participant_id>/<prompt_id>.wav
  - This script walks the raw audio tree, resolves each prompt ID to its best
    matching audio file, and produces a JSON Lines manifest plus helpful summary
    metadata for training / QA.

Output:
  A JSONL file (one JSON object per line) with fields:
    {
      "id": "<prompt_id>",
      "intent": "<canonical_intent>",
      "text": "<spoken text>",
      "slot_type": "...",
      "slot_value": "...",
      "audio_path": "data/raw/P01/nav_home_1.wav" | null,
      "participant_id": "P01" | null,
      "missing": false | true,
      "duplicates": ["path1","path2"...]   # only if >1 match
    }

Features:
  - Supports comments (# ...) in the CSV.
  - Validates duplicate prompt IDs.
  - Detects multiple candidate audio files per prompt (records them).
  - Provides summary stats (resolution rate, missing counts, duplicates).
  - Optional strict mode (--fail-on-missing) to exit non-zero if coverage incomplete.

Typical usage:
  python -m src.utils.build_audio_manifest \
      --prompts prompts_lean.csv \
      --raw-root data/raw \
      --out data/lean_dataset/audio_manifest.jsonl

Then you can use this manifest for feature extraction by treating "audio_path"
as the file column and "intent" as the label column.

Author: Auto-generated engineering utility
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Prompt:
    pid: str
    text: str
    intent: str
    slot_type: Optional[str]
    slot_value: Optional[str]

@dataclass
class ManifestEntry:
    id: str
    intent: str
    text: str
    slot_type: Optional[str]
    slot_value: Optional[str]
    audio_path: Optional[str]
    participant_id: Optional[str]
    missing: bool
    duplicates: List[str]


# -----------------------------------------------------------------------------
# CSV Parsing
# -----------------------------------------------------------------------------

def read_prompts_csv(path: str,
                     id_col: str = "id",
                     text_col: str = "text",
                     intent_col: str = "canonical_intent",
                     slot_type_col: str = "slot_type",
                     slot_value_col: str = "slot_value") -> List[Prompt]:
    """
    Parse the lean prompts CSV (with comment lines allowed).
    Returns list of Prompt objects.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")

    prompts: List[Prompt] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda l: not l.strip().startswith("#"), f))
        required = [id_col, text_col, intent_col]
        for col in required:
            if reader.fieldnames is None or col not in reader.fieldnames:
                raise ValueError(f"Required column '{col}' not found in CSV headers: {reader.fieldnames}")

        for line_num, row in enumerate(reader, start=2):
            pid = (row.get(id_col) or "").strip()
            if not pid:
                continue
            text = (row.get(text_col) or "").strip()
            intent = (row.get(intent_col) or "").strip()
            if not intent:
                continue
            slot_type = (row.get(slot_type_col) or "").strip() or None
            slot_value = (row.get(slot_value_col) or "").strip() or None

            if pid in seen_ids:
                # We allow duplicates but warn; could also raise if desired.
                print(f"[WARN] Duplicate prompt id '{pid}' at CSV line {line_num}", file=sys.stderr)
            else:
                seen_ids.add(pid)

            prompts.append(Prompt(
                pid=pid,
                text=text,
                intent=intent,
                slot_type=slot_type,
                slot_value=slot_value
            ))
    return prompts


# -----------------------------------------------------------------------------
# Audio Resolution
# -----------------------------------------------------------------------------

def index_audio_files(raw_root: str,
                      audio_ext: str) -> Dict[str, List[str]]:
    """
    Traverse the raw_root recursively and index audio by filename stem (without extension).
    Expected layout:
       raw_root/
         P01/
           nav_home_1.wav
         P02/
           nav_home_1.wav
    Returns mapping: <prompt_id> -> list of full paths
    """
    raw_path = Path(raw_root)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw audio root does not exist: {raw_root}")

    audio_index: Dict[str, List[str]] = defaultdict(list)
    ext_lower = audio_ext.lower()

    for path in raw_path.rglob(f"*{audio_ext}"):
        if not path.is_file():
            continue
        # Extract prompt_id candidate = filename without extension
        stem = path.stem
        # Example: nav_home_1.wav → nav_home_1
        audio_index[stem].append(str(path))

    # Also support case-insensitive extension match if user stores .WAV
    if audio_ext.lower() != audio_ext:
        for path in raw_path.rglob(f"*{audio_ext.upper()}"):
            if path.is_file():
                audio_index[path.stem].append(str(path))

    return audio_index


def resolve_prompt_audio(prompt: Prompt,
                         audio_index: Dict[str, List[str]],
                         raw_root: str) -> ManifestEntry:
    """
    Resolve a single prompt to zero/one/multiple candidate audio paths.
    If multiple matches: first one chosen as primary; all listed in duplicates.
    Participant extracted from path: raw_root/<participant>/<file>
    """
    candidates = audio_index.get(prompt.pid, [])
    candidates_sorted = sorted(candidates, key=lambda p: (len(p), p))  # deterministic
    chosen = candidates_sorted[0] if candidates_sorted else None

    participant_id = None
    if chosen:
        try:
            rel = os.path.relpath(chosen, raw_root)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                participant_id = parts[0]
        except Exception:
            participant_id = None

    return ManifestEntry(
        id=prompt.pid,
        intent=prompt.intent,
        text=prompt.text,
        slot_type=prompt.slot_type,
        slot_value=prompt.slot_value,
        audio_path=chosen,
        participant_id=participant_id,
        missing=(chosen is None),
        duplicates=candidates_sorted[1:] if len(candidates_sorted) > 1 else []
    )


# -----------------------------------------------------------------------------
# Manifest Writing
# -----------------------------------------------------------------------------

def write_manifest(entries: List[ManifestEntry], out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            rec = {
                "id": e.id,
                "intent": e.intent,
                "text": e.text,
                "slot_type": e.slot_type,
                "slot_value": e.slot_value,
                "audio_path": e.audio_path,
                "participant_id": e.participant_id,
                "missing": e.missing,
                "duplicates": e.duplicates
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize(entries: List[ManifestEntry]) -> Dict[str, Any]:
    total = len(entries)
    missing = sum(1 for e in entries if e.missing)
    duplicates = sum(1 for e in entries if len(e.duplicates) > 0)
    intents = {}
    for e in entries:
        intents[e.intent] = intents.get(e.intent, 0) + 1
    coverage = (total - missing) / total * 100 if total > 0 else 0.0
    return {
        "total_prompts": total,
        "resolved_audio": total - missing,
        "missing_audio": missing,
        "missing_pct": round(missing / total * 100, 2) if total else 0.0,
        "coverage_pct": round(coverage, 2),
        "prompts_with_duplicate_audio_candidates": duplicates,
        "unique_intents": len(intents)
    }


def write_summary(summary: Dict[str, Any], entries: List[ManifestEntry], out_summary: str):
    out_dir = os.path.dirname(out_summary)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save JSON summary
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also write a human-readable text summary
    text_path = os.path.splitext(out_summary)[0] + ".txt"
    lines = []
    lines.append("Audio Manifest Summary")
    lines.append("=" * 30)
    lines.append(f"Total prompts           : {summary['total_prompts']}")
    lines.append(f"Resolved audio          : {summary['resolved_audio']}")
    lines.append(f"Missing audio           : {summary['missing_audio']} ({summary['missing_pct']}%)")
    lines.append(f"Coverage percentage     : {summary['coverage_pct']}%")
    lines.append(f"Prompts w/ duplicates   : {summary['prompts_with_duplicate_audio_candidates']}")
    lines.append(f"Unique intents          : {summary['unique_intents']}")
    lines.append("")
    lines.append("Missing Prompt IDs (if any):")
    missing_ids = [e.id for e in entries if e.missing]
    if missing_ids:
        for mid in missing_ids[:50]:
            lines.append(f"  - {mid}")
        if len(missing_ids) > 50:
            lines.append(f"  ... ({len(missing_ids)-50} more)")
    else:
        lines.append("  None 🎉")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build audio manifest from lean prompts and participant-scoped recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--prompts", type=str, default="prompts_lean.csv",
                  help="Path to lean prompts CSV (with canonical_intent).")
    p.add_argument("--raw-root", type=str, default="data/raw",
                  help="Root directory containing participant subfolders.")
    p.add_argument("--out", type=str, default="data/lean_dataset/audio_manifest.jsonl",
                  help="Output manifest JSONL path.")
    p.add_argument("--audio-ext", type=str, default=".wav",
                  help="Audio file extension to match.")
    p.add_argument("--fail-on-missing", action="store_true",
                  help="Exit with non-zero status if any prompt is missing audio.")
    p.add_argument("--canonical-intent-column", type=str, default="canonical_intent",
                  help="Column name for canonical intent label.")
    return p.parse_args(argv)


def main():
    args = parse_args()

    print(f"[INFO] Reading prompts from: {args.prompts}")
    prompts = read_prompts_csv(
        args.prompts,
        id_col="id",
        text_col="text",
        intent_col=args.canonical_intent_column,
        slot_type_col="slot_type",
        slot_value_col="slot_value"
    )
    print(f"[INFO] Parsed {len(prompts)} prompt rows.")

    print(f"[INFO] Indexing audio under: {args.raw_root}")
    # Ensure second argument is the audio extension string, not the full Namespace
    audio_index = index_audio_files(raw_root=args.raw_root, audio_ext=args.audio_ext)
    print(f"[INFO] Indexed {sum(len(v) for v in audio_index.values())} audio files "
          f"across {len(audio_index)} unique stems.")

    # Resolve each prompt
    manifest_entries: List[ManifestEntry] = []
    for pr in prompts:
        entry = resolve_prompt_audio(pr, audio_index, args.raw_root)
        manifest_entries.append(entry)

    # Write manifest
    print(f"[INFO] Writing manifest to: {args.out}")
    write_manifest(manifest_entries, args.out)

    # Summary
    summary = summarize(manifest_entries)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    print(f"[INFO] Writing summary to: {summary_path}")
    write_summary(summary, manifest_entries, summary_path)

    print("\n---- Manifest Summary ----")
    for k, v in summary.items():
        print(f"{k:35s}: {v}")

    if args.fail_on_missing and summary["missing_audio"] > 0:
        print("[ERROR] Missing audio detected and --fail-on-missing set. Exiting with code 1.")
        sys.exit(1)

    print("\n[OK] Audio manifest build complete.")


if __name__ == "__main__":
    main()
