#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import pandas as pd
from collections import defaultdict

# ---------- CONFIG ----------
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["en", "de"]

DOMAIN_PREFIX = "(classificationName=linsearch:mapping)"

OUT_SPLIT = "domain_total_occurrences_by_split.csv"
OUT_TYPE_LANG = "domain_total_occurrences_by_type_lang.csv"
# ---------------------------

# ---- Domain codes (abbreviations only; used for matching) ----
DOMAIN_CODES = [
    "arc","bau","che","elt","fer","his","inf","lin","lit","mat","oek","pae","phi",
    "phy","sow","tec","ver","ber","bio","cet","geo","hor","jur","mas","med","meda",
    "rel","rest","spo"
]

# Regex: match either "linsearch:abc" or bare "abc" (word-bounded), case-insensitive
CURIE_PATTERNS = {c: re.compile(rf"\blinsearch:{re.escape(c)}\b", re.IGNORECASE) for c in DOMAIN_CODES}
CODE_PATTERNS  = {c: re.compile(rf"\b{re.escape(c)}\b", re.IGNORECASE)            for c in DOMAIN_CODES}


def get_folder_input(prompt_text, default=None):
    """Prompt user for a folder path with optional default fallback."""
    if len(sys.argv) > 1 and prompt_text.startswith("Enter the path to your data"):
        path = sys.argv[1]
        return path
    path = input(prompt_text).strip()
    if not path and default:
        return default
    return path


# --- Ask for user inputs ---
ROOT = get_folder_input("Enter the path to your data folder (e.g., ./library-records-dataset/data): ")
if not os.path.isdir(ROOT):
    raise ValueError(f"The provided data folder does not exist: {ROOT}")

OUTPUT_DIR = get_folder_input("Enter output directory (press Enter to use current working directory): ",
                              os.getcwd())
if not os.path.isdir(OUTPUT_DIR):
    raise ValueError(f"The provided output directory does not exist: {OUTPUT_DIR}")


# Map split names to physical paths (mirrors your original structure)
SPLIT_DIRS = {
    "train": [os.path.join(ROOT, "train")],
    "dev":   [os.path.join(ROOT, "dev")],
    "test":  [os.path.join(ROOT, "test", "gold-standard-testset"), os.path.join(ROOT, "test")],
}


def extract_domain_occurrences_from_subject_value(val: str) -> int:
    """
    Count how many domain codes appear in a subject string (value must start with DOMAIN_PREFIX).
    Each matched code contributes 1 to the count (even if multiple domains appear in one value).
    """
    if not (isinstance(val, str) and val.startswith(DOMAIN_PREFIX)):
        return 0

    count = 0
    for code in DOMAIN_CODES:
        if CURIE_PATTERNS[code].search(val) or CODE_PATTERNS[code].search(val):
            count += 1
    return count


def count_file_domain_occurrences(path: str) -> int:
    """
    Return total domain occurrences in one JSON-LD file (sum across all subject values, all items).
    """
    total = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0

    graph = data.get("@graph", [])
    if not isinstance(graph, list):
        return 0

    for item in graph:
        subj = item.get("subject")
        if subj is None:
            continue
        if isinstance(subj, list):
            for s in subj:
                if isinstance(s, str):
                    total += extract_domain_occurrences_from_subject_value(s)
        elif isinstance(subj, str):
            total += extract_domain_occurrences_from_subject_value(subj)

    return total


# --- Aggregations ---
# 1) Per split
occ_by_split = defaultdict(int)  # split -> total occurrences

# 2) Per (Type, Lang) across all splits
occ_by_type_lang = defaultdict(int)  # (type, lang) -> total occurrences

for split, base_dirs in SPLIT_DIRS.items():
    base_dir = next((d for d in base_dirs if os.path.isdir(d)), None)
    if not base_dir:
        continue

    for doc_type in DOC_TYPES:
        for lang in LANGS:
            lang_dir = os.path.join(base_dir, doc_type, lang)
            if not os.path.isdir(lang_dir):
                continue

            for fname in os.listdir(lang_dir):
                if not fname.endswith(".jsonld"):
                    continue
                fpath = os.path.join(lang_dir, fname)
                occ = count_file_domain_occurrences(fpath)

                # Aggregate
                occ_by_split[split] += occ
                occ_by_type_lang[(doc_type, lang)] += occ


# --- Build CSV #1: totals per split + overall ---
split_rows = []
overall_total = 0
for split in ["train", "dev", "test"]:
    total = occ_by_split.get(split, 0)
    overall_total += total
    split_rows.append({"Split": split, "Total Occurrences": total})

split_rows.append({"Split": "ALL", "Total Occurrences": overall_total})

df_split = pd.DataFrame(split_rows)
df_split_path = os.path.join(OUTPUT_DIR, OUT_SPLIT)
df_split.to_csv(df_split_path, index=False)


# --- Build CSV #2: totals per (Type, Lang) + marginal totals ---
# Base grid (Type × Lang)
rows = []
for doc_type in DOC_TYPES:
    for lang in LANGS:
        rows.append({
            "Type": doc_type,
            "Lang": lang,
            "Total Occurrences": occ_by_type_lang.get((doc_type, lang), 0)
        })

# Marginal totals: per Type (Lang=ALL), per Lang (Type=ALL)
# Per Type totals
for doc_type in DOC_TYPES:
    s = sum(occ_by_type_lang.get((doc_type, l), 0) for l in LANGS)
    rows.append({"Type": doc_type, "Lang": "ALL", "Total Occurrences": s})

# Per Lang totals
for lang in LANGS:
    s = sum(occ_by_type_lang.get((t, lang), 0) for t in DOC_TYPES)
    rows.append({"Type": "ALL", "Lang": lang, "Total Occurrences": s})

# Overall (ALL, ALL) — mirrors Split=ALL but kept here for completeness
overall_type_lang = sum(v for v in occ_by_type_lang.values())
rows.append({"Type": "ALL", "Lang": "ALL", "Total Occurrences": overall_type_lang})

df_type_lang = pd.DataFrame(rows).sort_values(["Type", "Lang"]).reset_index(drop=True)
df_type_lang_path = os.path.join(OUTPUT_DIR, OUT_TYPE_LANG)
df_type_lang.to_csv(df_type_lang_path, index=False)
