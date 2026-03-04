#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import pandas as pd
from collections import defaultdict

# ---------------- CONFIG ----------------
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["en", "de"]

# Output file
OUT_CSV = "yearwise_record_counts_by_split.csv"
# ---------------------------------------


def get_folder_input(prompt_text, default=None):
    """Prompt user for a folder path with optional default fallback or CLI arg."""
    if len(sys.argv) > 1 and prompt_text.startswith("Enter the path to your data"):
        return sys.argv[1]
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


# Map split names to physical paths (mirrors prior scripts)
SPLIT_DIRS = {
    "train": [os.path.join(ROOT, "train")],
    "dev":   [os.path.join(ROOT, "dev")],
    "test":  [os.path.join(ROOT, "test", "gold-standard-testset"), os.path.join(ROOT, "test")],
}


# ---------- Year extraction helpers ----------

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b")  # safe range 1500..2100

def _coerce_year(val):
    """
    Try to extract a 4-digit year from various JSON-LD value shapes:
    - int (e.g., 1998)
    - string '1998' or '1998-05-12' or ISO-like
    - dict with '@value': '1998-05-12' or {'@value': 1998}
    Returns int year or None.
    """
    if val is None:
        return None
    # dict with @value
    if isinstance(val, dict) and "@value" in val:
        return _coerce_year(val["@value"])
    # list: take first sensible year found
    if isinstance(val, list):
        for v in val:
            y = _coerce_year(v)
            if isinstance(y, int):
                return y
        return None
    # integer
    if isinstance(val, int):
        if 1500 <= val <= 2100:
            return val
        return None
    # string
    if isinstance(val, str):
        # common simple case: '1998' or '1998-..'
        # try exact int first
        if val.isdigit() and len(val) == 4:
            try:
                y = int(val)
                return y if 1500 <= y <= 2100 else None
            except Exception:
                pass
        # search anywhere in string
        m = YEAR_RE.search(val)
        if m:
            y = int(m.group(0))
            return y if 1500 <= y <= 2100 else None
    return None


def find_issued_year_from_graph(graph):
    """
    Given a JSON-LD @graph (list), try to find an 'issued' year.
    We check common keys:
      - 'issued'
      - 'dcterms:issued'
      - 'dc:date' / 'dcterms:date'
      - 'datePublished' (schema.org)
      - any key that endswith ':issued'
    Returns int year or None.
    """
    CANDIDATE_KEYS = [
        "issued", "dcterms:issued", "dc:date", "dcterms:date",
        "datePublished", "publicationDate", "rda:dateOfPublication",
    ]

    # Pass 1: exact candidate keys
    for item in graph:
        if not isinstance(item, dict):
            continue
        for k in CANDIDATE_KEYS:
            if k in item:
                y = _coerce_year(item.get(k))
                if isinstance(y, int):
                    return y

    # Pass 2: any key that ends with ':issued'
    for item in graph:
        if not isinstance(item, dict):
            continue
        for k, v in item.items():
            if isinstance(k, str) and k.endswith(":issued"):
                y = _coerce_year(v)
                if isinstance(y, int):
                    return y

    # Pass 3: try a broader sweep over date-like keys
    for item in graph:
        if not isinstance(item, dict):
            continue
        for k, v in item.items():
            if not isinstance(k, str):
                continue
            # heuristic: keys with 'date' or 'issued' in them
            lk = k.lower()
            if "issued" in lk or "date" in lk or "publication" in lk:
                y = _coerce_year(v)
                if isinstance(y, int):
                    return y

    return None


def extract_issued_year_from_file(path):
    """Open JSON-LD file, find @graph, then extract an issued year if available."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    graph = data.get("@graph", [])
    if not isinstance(graph, list):
        return None

    return find_issued_year_from_graph(graph)


# -------------- Aggregation --------------

# counts per (split, year)
year_counts_by_split = defaultdict(int)

# also keep overall counts per year
year_counts_overall = defaultdict(int)

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
                y = extract_issued_year_from_file(fpath)
                if isinstance(y, int):
                    year_counts_by_split[(split, y)] += 1
                    year_counts_overall[y] += 1


# -------------- Build CSV --------------

# Collect all years that appear anywhere
all_years = sorted(set(y for (_, y) in year_counts_by_split.keys()) | set(year_counts_overall.keys()))

rows = []

# Per split rows
for split in ["train", "dev", "test"]:
    for y in all_years:
        rows.append({
            "Split": split,
            "Year": y,
            "Count": year_counts_by_split.get((split, y), 0),
        })

# Overall rows
for y in all_years:
    rows.append({
        "Split": "ALL",
        "Year": y,
        "Count": year_counts_overall.get(y, 0),
    })

df = pd.DataFrame(rows).sort_values(["Split", "Year"]).reset_index(drop=True)

# Save
out_path = os.path.join(OUTPUT_DIR, OUT_CSV)
df.to_csv(out_path, index=False)
