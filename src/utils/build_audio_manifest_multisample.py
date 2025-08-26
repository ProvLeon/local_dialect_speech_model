#!/usr/bin/env python3
"""
build_audio_manifest_multisample.py

Purpose
-------
Build a JSONL audio manifest that enumerates EVERY available (participant, prompt_id, sample_number)
audio instance, supporting multi-sample canonical naming:

  data/raw/<participant>/<prompt_id>.wav          (legacy sample 1 only)
  data/raw/<participant>/<prompt_id>_s01.wav      (preferred explicit sample 1)
  data/raw/<participant>/<prompt_id>_s02.wav
  data/raw/<participant>/<prompt_id>_s03.wav
  ...

Each physical audio file becomes one manifest row so downstream feature extraction / training
can treat each instance independently, increasing dataset size.

Differences vs original build_audio_manifest.py
------------------------------------------------
1. Produces one entry per (participant, prompt_id, sample_number) instead of collapsing into
   a single canonical path per prompt.
2. Distinguishes sample numbers explicitly.
3. Includes a synthetic "instance_id" field that is guaranteed unique and stable given current files:
       instance_id = "<prompt_id>__<participant_id>__sXX"
4. Generates summary statistics about multi-sample distribution.
5. Ignores "duplicates" concept because each file becomes its own entry.

Expected Naming
---------------
Recorder (after multi-sample patch) creates:
  - Always: <prompt_id>_sNN.wav for every recorded sample (NN = 01, 02, ...)
  - Additionally (backward compatibility): <prompt_id>.wav for sample 1 (only if not already present).

This script:
  - Treats <prompt_id>_sNN.wav as authoritative for sample NN.
  - If ONLY <prompt_id>.wav exists (no _s01 variant), it is treated as sample 1.
  - If both exist, <prompt_id>_s01.wav is used and the bare <prompt_id>.wav is ignored (to avoid duplicates).

Manifest Row Schema
-------------------
{
  "instance_id": "nav_home_1__P01__s02",
  "base_id": "nav_home_1",
  "participant_id": "P01",
  "sample_number": 2,
  "audio_path": "data/raw/P01/nav_home_1_s02.wav",
  "intent": "navigation.home",
  "text": "navigate home",
  "slot_type": null,
  "slot_value": null
}

Summary JSON (path: <out>_summary.json):
{
  "total_prompts": 109,
  "participants": 2,
  "total_audio_instances": 436,
  "prompts_with_any_audio": 109,
  "coverage_pct": 100.0,
  "avg_samples_per_prompt_per_participant": 2.0,
  "min_samples_per_prompt_participant": 2,
  "max_samples_per_prompt_participant": 2
}

CLI
---
python -m src.utils.build_audio_manifest_multisample \
    --prompts prompts_lean.csv \
    --raw-root data/raw \
    --out data/lean_dataset/audio_manifest_multisample.jsonl

Author: Generated utility
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any, Set, DefaultDict
from collections import defaultdict, Counter


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class Prompt:
    pid: str
    text: str
    intent: str
    slot_type: Optional[str]
    slot_value: Optional[str]


@dataclass
class SampleEntry:
    instance_id: str
    base_id: str
    participant_id: str
    sample_number: int
    audio_path: str
    intent: str
    text: str
    slot_type: Optional[str]
    slot_value: Optional[str]


# -----------------------------------------------------------------------------
# CSV Parsing
# -----------------------------------------------------------------------------

def read_prompts_csv(path: str,
                     id_col: str = "id",
                     text_col: str = "text",
                     intent_col: str = "canonical_intent",
                     slot_type_col: str = "slot_type",
                     slot_value_col: str = "slot_value") -> Dict[str, Prompt]:
    """
    Parse lean prompts CSV (ignoring comment lines starting with '#').
    Returns mapping: prompt_id -> Prompt
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")

    prompts: Dict[str, Prompt] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda l: not l.strip().startswith("#"), f))
        if reader.fieldnames is None:
            raise ValueError("No CSV headers found.")

        required = [id_col, text_col, intent_col]
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Required column '{col}' missing from CSV headers: {reader.fieldnames}")

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
            if pid in prompts:
                print(f"[WARN] Duplicate prompt id '{pid}' at CSV line {line_num}", file=sys.stderr)
            prompts[pid] = Prompt(
                pid=pid,
                text=text,
                intent=intent,
                slot_type=slot_type,
                slot_value=slot_value
            )
    return prompts


# -----------------------------------------------------------------------------
# Audio Indexing (Multi-Sample Aware)
# -----------------------------------------------------------------------------

BARE_RE = re.compile(r"^(?P<pid>[A-Za-z0-9_-]+)$")
SAMPLE_RE = re.compile(r"^(?P<pid>[A-Za-z0-9_-]+)_s(?P<sn>\d{2})$")


def index_audio_multisample(raw_root: str,
                            audio_ext: str = ".wav") -> Dict[Tuple[str, str, int], str]:
    """
    Walk raw_root and index audio files by (prompt_id, participant_id, sample_number).
    Prefers explicit <pid>_s01.wav over <pid>.wav when both present.

    Returns mapping:
        (pid, participant_id, sample_number) -> filepath
    """
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw audio root does not exist: {raw_root}")

    # Temporary store to handle conflict resolution
    temp_map: Dict[Tuple[str, str, int], Dict[str, str]] = defaultdict(dict)
    participants: List[Path] = [p for p in root.iterdir() if p.is_dir()]

    for participant_dir in participants:
        participant_id = participant_dir.name
        for file in participant_dir.rglob(f"*{audio_ext}"):
            if not file.is_file():
                continue
            stem = file.stem
            m_sample = SAMPLE_RE.match(stem)
            if m_sample:
                pid = m_sample.group("pid")
                sn = int(m_sample.group("sn"))
                key = (pid, participant_id, sn)
                # Mark explicit sample variant
                temp_map[key]["explicit"] = str(file)
                continue
            m_bare = BARE_RE.match(stem)
            if m_bare:
                pid = m_bare.group("pid")
                sn = 1
                key = (pid, participant_id, sn)
                # Only record if no explicit variant captured
                temp_map[key].setdefault("bare", str(file))
                continue
            # ignore unrelated files

    # Resolve preference: explicit > bare
    resolved: Dict[Tuple[str, str, int], str] = {}
    for key, variants in temp_map.items():
        if "explicit" in variants:
            resolved[key] = variants["explicit"]
        else:
            resolved[key] = variants["bare"]

    return resolved


# -----------------------------------------------------------------------------
# Manifest Construction
# -----------------------------------------------------------------------------

def build_entries(prompts: Dict[str, Prompt],
                  audio_index: Dict[Tuple[str, str, int], str],
                  raw_root: str) -> List[SampleEntry]:
    """
    Create SampleEntry list from indexed audio.
    """
    entries: List[SampleEntry] = []
    for (pid, participant_id, sample_number), path in sorted(audio_index.items()):
        pr = prompts.get(pid)
        if not pr:
            # Audio for prompt id not in CSV (could optionally warn)
            print(f"[WARN] Audio found for unknown prompt id '{pid}' (participant={participant_id})", file=sys.stderr)
            continue
        instance_id = f"{pid}__{participant_id}__s{sample_number:02d}"
        entries.append(SampleEntry(
            instance_id=instance_id,
            base_id=pid,
            participant_id=participant_id,
            sample_number=sample_number,
            audio_path=path,
            intent=pr.intent,
            text=pr.text,
            slot_type=pr.slot_type,
            slot_value=pr.slot_value
        ))
    return entries


def write_manifest(entries: Iterable[SampleEntry], out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            rec = {
                "instance_id": e.instance_id,
                "base_id": e.base_id,
                "participant_id": e.participant_id,
                "sample_number": e.sample_number,
                "audio_path": e.audio_path,
                "intent": e.intent,
                "text": e.text,
                "slot_type": e.slot_type,
                "slot_value": e.slot_value
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def summarize(entries: List[SampleEntry],
              prompts: Dict[str, Prompt]) -> Dict[str, Any]:
    total_prompts = len(prompts)
    # Prompts covered by at least one audio instance
    covered_prompts: Set[str] = {e.base_id for e in entries}
    coverage_pct = (len(covered_prompts) / total_prompts * 100.0) if total_prompts else 0.0

    # Participants
    participants = {e.participant_id for e in entries}

    # Samples per prompt per participant
    sppp: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for e in entries:
        sppp[(e.base_id, e.participant_id)].append(e.sample_number)

    samples_counts = [len(v) for v in sppp.values()]
    if samples_counts:
        avg_samples = sum(samples_counts) / len(samples_counts)
        min_samples = min(samples_counts)
        max_samples = max(samples_counts)
    else:
        avg_samples = min_samples = max_samples = 0

    summary = {
        "total_prompts": total_prompts,
        "participants": len(participants),
        "participant_ids": sorted(participants),
        "total_audio_instances": len(entries),
        "prompts_with_any_audio": len(covered_prompts),
        "coverage_pct": round(coverage_pct, 2),
        "avg_samples_per_prompt_per_participant": round(avg_samples, 3),
        "min_samples_per_prompt_participant": min_samples,
        "max_samples_per_prompt_participant": max_samples,
    }
    return summary


def write_summary(summary: Dict[str, Any], out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Human readable
    txt_path = os.path.splitext(out_path)[0] + ".txt"
    lines = []
    lines.append("Multi-Sample Audio Manifest Summary")
    lines.append("=" * 40)
    for k, v in summary.items():
        if k == "participant_ids":
            continue
        lines.append(f"{k:35s}: {v}")
    if summary.get("participant_ids"):
        lines.append("\nParticipants:")
        for pid in summary["participant_ids"]:
            lines.append(f"  - {pid}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build multi-sample audio manifest (one row per participant/sample).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--prompts", type=str, default="prompts_lean.csv",
                   help="Path to lean prompts CSV.")
    p.add_argument("--raw-root", type=str, default="data/raw",
                   help="Root directory containing participant subfolders.")
    p.add_argument("--out", type=str, default="data/lean_dataset/audio_manifest_multisample.jsonl",
                   help="Output multi-sample manifest JSONL path.")
    p.add_argument("--audio-ext", type=str, default=".wav",
                   help="Audio extension to match.")
    p.add_argument("--canonical-intent-column", type=str, default="canonical_intent",
                   help="Column name for canonical intent label.")
    return p.parse_args(argv)


def main():
    args = parse_args()

    print(f"[INFO] Reading prompts: {args.prompts}")
    prompts = read_prompts_csv(
        args.prompts,
        id_col="id",
        text_col="text",
        intent_col=args.canonical_intent_column,
        slot_type_col="slot_type",
        slot_value_col="slot_value"
    )
    print(f"[INFO] Loaded {len(prompts)} prompts.")

    print(f"[INFO] Indexing multi-sample audio under: {args.raw_root}")
    audio_index = index_audio_multisample(args.raw_root, args.audio_ext)
    print(f"[INFO] Indexed {len(audio_index)} (prompt, participant, sample) combinations.")

    print(f"[INFO] Building entries...")
    entries = build_entries(prompts, audio_index, args.raw_root)
    print(f"[INFO] Built {len(entries)} manifest rows.")

    print(f"[INFO] Writing manifest: {args.out}")
    count = write_manifest(entries, args.out)
    print(f"[INFO] Wrote {count} rows.")

    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    summary = summarize(entries, prompts)
    print(f"[INFO] Writing summary: {summary_path}")
    write_summary(summary, summary_path)

    print("\n---- Multi-Sample Manifest Summary ----")
    for k, v in summary.items():
        if k == "participant_ids":
            continue
        print(f"{k}: {v}")

    print("\n[OK] Multi-sample manifest build complete.")


if __name__ == "__main__":
    main()
