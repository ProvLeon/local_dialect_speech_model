#!/usr/bin/env python3
"""
backfill_sample_copies.py

Purpose
-------
Create canonical multi-sample copies for previously recorded long-form audio files.

Context
-------
Original (legacy) recorder behavior:
  - Created ONLY sample 1 canonical file:
        data/raw/<participant>/<prompt_id>.wav
  - Additional samples (s02, s03, ...) were saved ONLY as long-form filenames:
        <intent>_<safe_text>_sNN_<timestamp>.wav
    where:
        intent      = canonical_intent from CSV
        safe_text   = sanitized + truncated prompt text (see sanitize_text)
        NN          = zero-padded sample number
        timestamp   = epoch seconds

Desired current state:
  - For every recorded sample n (>=2) we want a canonical copy:
        data/raw/<participant>/<prompt_id>_sNN.wav
  - (Optionally) also ensure explicit _s01 copy exists via --create-s01-from-bare
    by copying <prompt_id>.wav -> <prompt_id>_s01.wav when missing.

This script scans a SOURCE directory containing participant subdirectories
with long-form recordings and copies (or symlinks / hardlinks) them into a
DESTINATION "raw" directory under canonical names.

Mapping Problem
---------------
We must map each long-form filename back to its prompt_id. The direct key is
(intent, safe_text). But drift can occur (case changes, minor edits) so we
offer multiple fallback strategies:

Resolution Order (when enabled):
  1. Exact (intent, safe_text)
  2. Case-insensitive
  3. Prefix fallback (safe_text startswith/endswith other within same intent)
  4. Fuzzy (SequenceMatcher ratio within same intent) above threshold
  5. Global safe_text unique (intent-agnostic) final fallback
If none resolves, file is counted as skipped_no_mapping.

CLI Flags enable / disable these stages.

Key Definitions
---------------
  safe_text = sanitize_text(original_prompt_text)
  sanitize_text:
     - Remove non-word / non-space / non-hyphen chars
     - Collapse groups of hyphen/whitespace to single underscore
     - Truncate to 20 characters
     (Must match recorder's logic)

Example Long-form Filename:
  add_address_Fa_address_foforɔ_bi_s02_1756188768.wav
    intent    = add_address
    safe_text = Fa_address_foforɔ_bi
    sample    = 02
    ts        = 1756188768

By default, sample 1 long-form files (_s01_) are ignored (we assume canonical
<prompt_id>.wav already exists). You can still force creation of an explicit
_s01 copy via --create-s01-from-bare.

Workflow
--------
1. Run a dry run:
   python -m src.utils.backfill_sample_copies \
       --prompts prompts_lean.csv \
       --source-root data/recordings \
       --raw-root data/raw \
       --dry-run --case-insensitive --prefix-fallback --fuzzy --global-safe-fallback \
       --debug-mapping --verbose
2. Inspect stats & any WARN lines.
3. Run without --dry-run (maybe add --copy-mode symlink if you prefer space savings).
4. Build multi-sample manifest from data/raw afterwards:
   python -m src.utils.build_audio_manifest_multisample --prompts prompts_lean.csv --raw-root data/raw --out data/lean_dataset/audio_manifest_multisample.jsonl

Stats Reported
--------------
  long_files_found
  created
  skipped_exists
  skipped_sample1
  skipped_ambiguous
  skipped_no_mapping
  created_s01_from_bare
  resolved_casefold
  resolved_prefix
  resolved_fuzzy
  resolved_safe_global

Copy Modes
----------
--copy-mode copy     (binary copy, default)
--copy-mode symlink  (create symlink to source file)
--copy-mode hardlink (create hard link; same inode if FS permits)

Safety
------
Never overwrites existing canonical files unless --overwrite is given.

Author
------
Consolidated & extended engineering utility.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable
from difflib import SequenceMatcher
import shutil

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("backfill_samples")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
@dataclass
class PromptRow:
    pid: str
    intent: str
    text: str
    slot_type: Optional[str]
    slot_value: Optional[str]
    safe_text: str

@dataclass
class Stats:
    participant: str
    long_files_found: int = 0
    created: int = 0
    skipped_exists: int = 0
    skipped_sample1: int = 0
    skipped_ambiguous: int = 0
    skipped_no_mapping: int = 0
    created_s01_from_bare: int = 0
    resolved_casefold: int = 0
    resolved_prefix: int = 0
    resolved_fuzzy: int = 0
    resolved_safe_global: int = 0

    def add(self, other: "Stats"):
        for f in self.__dataclass_fields__:
            if f == "participant":
                continue
            setattr(self, f, getattr(self, f) + getattr(other, f))

# -----------------------------------------------------------------------------
# Sanitization (MUST match recorder)
# -----------------------------------------------------------------------------
SANITIZE_RE_NON = re.compile(r"[^\w\s-]", re.UNICODE)
SANITIZE_RE_WS = re.compile(r"[-\s]+")

def sanitize_text(text: str) -> str:
    t = SANITIZE_RE_NON.sub("", text)
    t = SANITIZE_RE_WS.sub("_", t)
    return t[:20]

# -----------------------------------------------------------------------------
# Long-form filename parsing
# Pattern: <intent>_<safe_text>_sNN_<timestamp>.wav
# -----------------------------------------------------------------------------
LONG_FORM_RE = re.compile(
    r"^(?P<intent>[A-Za-z0-9_.-]+)_(?P<safe>.+)_s(?P<sample>\d{2})_(?P<ts>\d+)$"
)

def parse_long_form(path: Path):
    m = LONG_FORM_RE.match(path.stem)
    if not m:
        return None
    return {
        "intent": m.group("intent"),
        "safe_text": m.group("safe"),
        "sample_number": int(m.group("sample")),
        "timestamp": m.group("ts")
    }

# -----------------------------------------------------------------------------
# Prompt CSV Loading
# -----------------------------------------------------------------------------
def load_prompts(prompts_path: str,
                 id_col="id",
                 text_col="text",
                 intent_col="canonical_intent",
                 slot_type_col="slot_type",
                 slot_value_col="slot_value") -> Dict[str, PromptRow]:
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    prompts: Dict[str, PromptRow] = {}
    with open(prompts_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda l: not l.strip().startswith("#"), f))
        if not reader.fieldnames:
            raise ValueError("CSV missing headers.")
        for col in (id_col, text_col, intent_col):
            if col not in reader.fieldnames:
                raise ValueError(f"Required column '{col}' not in {reader.fieldnames}")

        for row in reader:
            pid = (row.get(id_col) or "").strip()
            if not pid:
                continue
            intent = (row.get(intent_col) or "").strip()
            if not intent:
                continue
            text = (row.get(text_col) or "").strip()
            slot_t = (row.get(slot_type_col) or "").strip() or None
            slot_v = (row.get(slot_value_col) or "").strip() or None
            prompts[pid] = PromptRow(
                pid=pid,
                intent=intent,
                text=text,
                slot_type=slot_t,
                slot_value=slot_v,
                safe_text=sanitize_text(text)
            )
    return prompts

# -----------------------------------------------------------------------------
# Indices for Resolution
# -----------------------------------------------------------------------------
def build_indices(prompts: Dict[str, PromptRow]):
    # exact (intent, safe) -> [pid]
    exact: Dict[Tuple[str, str], List[str]] = {}
    # lowered
    lowered: Dict[Tuple[str, str], List[str]] = {}
    # per intent all safe variants (original case)
    per_intent_all: Dict[str, List[Tuple[str, List[str]]]] = {}
    # per intent unambiguous lowered for fuzzy
    per_intent_unambig_lower: Dict[str, List[Tuple[str, str]]] = {}
    # global safe lowered -> [pid] for global unique fallback
    global_safe: Dict[str, List[str]] = {}

    for pid, row in prompts.items():
        key = (row.intent, row.safe_text)
        exact.setdefault(key, []).append(pid)

    for (intent, safe), pids in exact.items():
        lower_key = (intent.lower(), safe.lower())
        lowered.setdefault(lower_key, []).append(pids[0] if len(pids) == 1 else ",".join(pids))
        per_intent_all.setdefault(intent.lower(), []).append((safe, pids))
        if len(pids) == 1:
            per_intent_unambig_lower.setdefault(intent.lower(), []).append((safe.lower(), pids[0]))
            global_safe.setdefault(safe.lower(), []).append(pids[0])

    return exact, lowered, per_intent_all, per_intent_unambig_lower, global_safe

# -----------------------------------------------------------------------------
# File copy helpers
# -----------------------------------------------------------------------------
def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def materialize(dest: Path, src: Path, mode: str, overwrite: bool):
    if dest.exists():
        if not overwrite:
            return "exists"
        dest.unlink()

    ensure_parent(dest)
    if mode == "copy":
        shutil.copy2(src, dest)
    elif mode == "symlink":
        # Relative symlink for portability
        rel = os.path.relpath(src, dest.parent)
        os.symlink(rel, dest)
    elif mode == "hardlink":
        os.link(src, dest)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")
    return "created"

# -----------------------------------------------------------------------------
# Intent Normalization
# -----------------------------------------------------------------------------
def normalize_intent(verbose_intent: str) -> str:
    """
    Extract canonical intent from verbose old intent names.

    Pattern: <canonical_intent>_<first_word>_<maybe_more>
    Examples:
        add_address_Fa_address -> add_address
        apply_filter_Fa_filter_fa -> apply_filter
        cancel_order_Gyae_order -> cancel_order
    """
    # Known canonical intents from prompts_lean.csv (hardcoded for robustness)
    canonical_intents = {
        'add_address', 'add_to_cart', 'apply_coupon', 'apply_filter',
        'cancel_order', 'change_color', 'change_quantity', 'change_size',
        'checkout', 'clear_cart', 'clear_filter', 'confirm_order', 'continue',
        'disable_order_updates', 'disable_price_alert', 'enable_order_updates',
        'enable_price_alert', 'exchange_item', 'go_back', 'go_home', 'help',
        'make_payment', 'order_not_arrived', 'refund_status', 'remove_address',
        'remove_coupon', 'remove_from_cart', 'save_for_later', 'select_color',
        'select_size', 'set_default_address', 'show_cart', 'show_description',
        'sort_items', 'start_live_chat', 'track_order'
    }

    # Try progressively shorter prefixes
    parts = verbose_intent.split('_')
    for i in range(len(parts), 0, -1):
        candidate = '_'.join(parts[:i])
        if candidate in canonical_intents:
            return candidate

    # Fallback: return original
    return verbose_intent

# -----------------------------------------------------------------------------
# Resolver
# -----------------------------------------------------------------------------
def resolve_pid(intent: str,
                safe_text: str,
                indices,
                *,
                enable_casefold: bool,
                enable_prefix: bool,
                enable_fuzzy: bool,
                fuzzy_threshold: float,
                enable_global_safe: bool,
                debug: bool,
                trace: List[str]) -> Optional[str]:
    exact, lowered, per_intent_all, per_intent_unambig_lower, global_safe = indices

    # 1. Exact match
    pids = exact.get((intent, safe_text))
    if pids:
        if len(pids) == 1:
            trace.append("exact")
            return pids[0]
        trace.append("ambiguous_exact")
        return None
    trace.append("miss_exact")

    # 2. Intent normalization (verbose old intent -> canonical)
    normalized_intent = normalize_intent(intent)
    if normalized_intent != intent:
        pids = exact.get((normalized_intent, safe_text))
        if pids:
            if len(pids) == 1:
                trace.append("normalized")
                return pids[0]
            trace.append("ambiguous_normalized")
            return None
        trace.append("miss_normalized")

    # 3. Case-insensitive (using normalized intent)
    if enable_casefold:
        search_intent = normalized_intent if normalized_intent != intent else intent
        pids_ci = lowered.get((search_intent.lower(), safe_text.lower()))
        if pids_ci:
            # pids_ci could be list of strings; we stored single or comma-joined (rare)
            pid_candidates = pids_ci if isinstance(pids_ci, list) else [pids_ci]
            if len(pid_candidates) == 1 and "," not in pid_candidates[0]:
                trace.append("casefold")
                return pid_candidates[0]
            trace.append("ambiguous_casefold")
            return None
        trace.append("miss_casefold")

    # 4. Prefix fallback (same intent)
    if enable_prefix:
        search_intent = normalized_intent if normalized_intent != intent else intent
        intent_l = search_intent.lower()
        target_l = safe_text.lower()
        hits = []
        for safe_variant, variant_pids in per_intent_all.get(intent_l, []):
            sv_l = safe_variant.lower()
            if target_l.startswith(sv_l) or sv_l.startswith(target_l):
                if len(variant_pids) == 1:
                    hits.append(variant_pids[0])
        hits = list(set(hits))
        if len(hits) == 1:
            trace.append("prefix")
            return hits[0]
        trace.append(f"miss_prefix({len(hits)})")

    # 5. Fuzzy (SequenceMatcher) within intent
    if enable_fuzzy:
        search_intent = normalized_intent if normalized_intent != intent else intent
        intent_l = search_intent.lower()
        candidates = per_intent_unambig_lower.get(intent_l, [])
        target_l = safe_text.lower()
        best_pid = None
        best_score = 0.0
        for safe_l, pid in candidates:
            ratio = SequenceMatcher(None, target_l, safe_l).ratio()
            if ratio > best_score:
                best_score = ratio
                best_pid = pid
        if best_pid and best_score >= fuzzy_threshold:
            trace.append(f"fuzzy({best_score:.3f})")
            return best_pid
        trace.append(f"miss_fuzzy({best_score:.3f})")

    # 6. Global safe unique
    if enable_global_safe:
        g = global_safe.get(safe_text.lower())
        if g and len(g) == 1:
            trace.append("global_safe")
            return g[0]
        trace.append("miss_global_safe")

    return None

# -----------------------------------------------------------------------------
# Participant Processing
# -----------------------------------------------------------------------------
def collect_long_form_files(source_participant_dir: Path, ext: str) -> List[Path]:
    files = []
    for p in source_participant_dir.glob(f"*{ext}"):
        if LONG_FORM_RE.match(p.stem):
            files.append(p)
    return files

def backfill_participant(participant_id: str,
                         source_dir: Path,
                         dest_dir: Path,
                         prompts: Dict[str, PromptRow],
                         indices,
                         *,
                         create_s01_from_bare: bool,
                         copy_mode: str,
                         overwrite: bool,
                         min_sample: int,
                         enable_casefold: bool,
                         enable_prefix: bool,
                         enable_fuzzy: bool,
                         fuzzy_threshold: float,
                         enable_global_safe: bool,
                         dry_run: bool,
                         debug_mapping: bool,
                         verbose: bool) -> Stats:
    stats = Stats(participant=participant_id)
    part_source = source_dir / participant_id
    part_dest = dest_dir / participant_id
    if not part_source.exists():
        LOG.warning(f"Participant '{participant_id}' missing in source root; skipping.")
        return stats

    # Optionally create s01 from bare
    if create_s01_from_bare:
        for pid in prompts:
            bare = part_dest / f"{pid}.wav"
            s01 = part_dest / f"{pid}_s01.wav"
            if bare.exists() and not s01.exists():
                if dry_run:
                    stats.created_s01_from_bare += 1
                    if verbose:
                        print(f"[DRY] Would create {s01.name} from {bare.name}")
                else:
                    try:
                        result = materialize(s01, bare, copy_mode, overwrite=False)
                        if result in ("created",):
                            stats.created_s01_from_bare += 1
                            if verbose:
                                print(f"[OK] Created {s01.name} from bare")
                    except Exception as e:
                        LOG.error(f"Failed to create s01 copy for {pid}: {e}")

    long_files = collect_long_form_files(part_source, ".wav")
    stats.long_files_found = len(long_files)

    for lf in long_files:
        meta = parse_long_form(lf)
        if not meta:
            continue
        sample = meta["sample_number"]
        if sample < min_sample:
            stats.skipped_sample1 += 1
            continue

        intent = meta["intent"]
        safe_text = meta["safe_text"]

        trace: List[str] = []
        pid = resolve_pid(
            intent, safe_text, indices,
            enable_casefold=enable_casefold,
            enable_prefix=enable_prefix,
            enable_fuzzy=enable_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
            enable_global_safe=enable_global_safe,
            debug=debug_mapping,
            trace=trace
        )

        if pid is None:
            stats.skipped_no_mapping += 1
            if verbose or debug_mapping:
                print(f"[WARN] No mapping: {lf.name} intent={intent} safe={safe_text} trace={'>'.join(trace)}")
            continue

        # Update resolution counters
        if "normalized" in trace: stats.resolved_prefix += 1  # normalized counts as prefix resolution
        elif "casefold" in trace: stats.resolved_casefold += 1
        elif any(t.startswith("prefix") for t in trace): stats.resolved_prefix += 1
        elif any(t.startswith("fuzzy(") for t in trace): stats.resolved_fuzzy += 1
        elif "global_safe" in trace: stats.resolved_safe_global += 1
        elif "exact" in trace:
            pass
        elif any(t.startswith("ambiguous") for t in trace):
            stats.skipped_ambiguous += 1
            continue

        # Destination canonical file
        dest_file = part_dest / f"{pid}_s{sample:02d}.wav"
        if dest_file.exists() and not overwrite:
            stats.skipped_exists += 1
            continue

        if dry_run:
            stats.created += 1
            if verbose:
                action = "Would overwrite" if dest_file.exists() else "Would create"
                print(f"[DRY] {action}: {dest_file.relative_to(dest_dir)}  <- {lf.relative_to(source_dir)}")
            continue

        try:
            result = materialize(dest_file, lf, copy_mode, overwrite=overwrite)
            if result == "created":
                stats.created += 1
                if verbose:
                    print(f"[OK] Created {dest_file.relative_to(dest_dir)} from {lf.name}")
            elif result == "exists":
                stats.skipped_exists += 1
        except Exception as e:
            LOG.error(f"Copy failed {lf} -> {dest_file}: {e}")

    return stats

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Backfill canonical multi-sample copies from long-form audio filenames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--prompts", type=str, default="prompts_lean.csv",
                    help="Lean prompts CSV path.")
    ap.add_argument("--source-root", type=str, default="data/recordings",
                    help="Root containing participant folders with LONG-FORM files.")
    ap.add_argument("--raw-root", type=str, default="data/raw",
                    help="Destination root for canonical copies.")
    ap.add_argument("--ext", type=str, default=".wav",
                    help="Audio file extension to process.")
    ap.add_argument("--participants", type=str, default="",
                    help="Comma-separated subset of participant IDs (default: all discovered).")
    ap.add_argument("--min-sample", type=int, default=2,
                    help="Minimum sample number in long-form files to backfill (2 means skip s01).")
    ap.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="copy",
                    help="Method used to create canonical copies.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show actions without writing anything.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing canonical _sNN files.")
    ap.add_argument("--create-s01-from-bare", action="store_true",
                    help="Create <prompt_id>_s01.wav if <prompt_id>.wav exists and _s01 missing.")
    # Fallback resolution flags
    ap.add_argument("--case-insensitive", action="store_true",
                    help="Enable case-insensitive mapping fallback.")
    ap.add_argument("--prefix-fallback", action="store_true",
                    help="Enable prefix/substring fallback within same intent.")
    ap.add_argument("--fuzzy", action="store_true",
                    help="Enable fuzzy SequenceMatcher fallback within same intent.")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.88,
                    help="Fuzzy ratio acceptance threshold.")
    ap.add_argument("--global-safe-fallback", action="store_true",
                    help="Enable intent-agnostic safe_text unique fallback.")
    ap.add_argument("--debug-mapping", action="store_true",
                    help="Print mapping trace for unmapped files.")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose per-file output.")
    return ap.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    LOG.info(f"Loading prompts: {args.prompts}")
    prompts = load_prompts(args.prompts)
    LOG.info(f"Loaded {len(prompts)} prompts.")

    LOG.info("Building indices...")
    indices = build_indices(prompts)
    exact = indices[0]
    ambiguous_exact = [k for k, v in exact.items() if len(v) > 1]
    if ambiguous_exact:
        LOG.warning(f"{len(ambiguous_exact)} (intent,safe_text) pairs ambiguous at exact stage.")

    source_root = Path(args.source_root)
    dest_root = Path(args.raw_root)

    if not source_root.exists():
        LOG.error(f"Source root does not exist: {source_root}")
        sys.exit(1)
    dest_root.mkdir(parents=True, exist_ok=True)

    if args.participants.strip():
        participants = [p.strip() for p in args.participants.split(",") if p.strip()]
    else:
        # Union of participant dirs in source root (folders only)
        participants = sorted([p.name for p in source_root.iterdir() if p.is_dir()])

    if not participants:
        LOG.warning("No participant directories found to process.")
        return

    LOG.info(f"Processing {len(participants)} participant(s)...")

    agg = Stats(participant="ALL")
    per_part: List[Stats] = []

    for pid in participants:
        if args.verbose:
            print(f"\n[PARTICIPANT] {pid}")
        st = backfill_participant(
            participant_id=pid,
            source_dir=source_root,
            dest_dir=dest_root,
            prompts=prompts,
            indices=indices,
            create_s01_from_bare=args.create_s01_from_bare,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
            min_sample=args.min_sample,
            enable_casefold=args.case_insensitive,
            enable_prefix=args.prefix_fallback,
            enable_fuzzy=args.fuzzy,
            fuzzy_threshold=args.fuzzy_threshold,
            enable_global_safe=args.global_safe_fallback,
            dry_run=args.dry_run,
            debug_mapping=args.debug_mapping,
            verbose=args.verbose
        )
        per_part.append(st)
        agg.add(st)

    # Summary
    print("\n----- Backfill Summary -----")
    print(f"Participants processed       : {len(per_part)}")
    for field in Stats.__dataclass_fields__:
        if field == "participant":
            continue
        print(f"{field}: {getattr(agg, field)}")
    print(f"Dry run                      : {args.dry_run}")
    print(f"Overwrite enabled            : {args.overwrite}")
    print(f"Create s01 from bare         : {args.create_s01_from_bare}")
    print(f"Case-insensitive             : {args.case_insensitive}")
    print(f"Prefix fallback              : {args.prefix_fallback}")
    print(f"Fuzzy                        : {args.fuzzy} (threshold={args.fuzzy_threshold})")
    print(f"Global safe fallback         : {args.global_safe_fallback}")
    print(f"Copy mode                    : {args.copy_mode}")
    print(f"Min sample                   : {args.min_sample}")

    if args.dry_run:
        print("\n[INFO] Dry run complete. Re-run without --dry-run to apply changes.")
    else:
        print("\n[OK] Backfill complete.")

if __name__ == "__main__":
    main()
