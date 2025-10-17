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

OUT_SPLIT_DOMAIN = "domain_occurrences_by_split_and_domain.csv"
OUT_TYPE_LANG_DOMAIN = "domain_occurrences_by_type_lang_and_domain.csv"
# ---------------------------

# ---- Domain codes + full forms (no IRIs) ----
DOMAIN_ROWS = [
    ("arc", "Architecture", "Architektur"),
    ("bau", "Civil engineering", "Bauwesen"),
    ("che", "Chemistry", "Chemie"),
    ("elt", "Electrical engineering", "Elektrotechnik"),
    ("fer", "Material science", "Werkstoffkunde"),
    ("his", "History", "Geschichte"),
    ("inf", "Computer Science", "Informatik"),
    ("lin", "Linguistics", "Sprachwissenschaften"),
    ("lit", "Literature Studies", "Literaturwissenschaften"),
    ("mat", "Mathematics", "Mathematik"),
    ("oek", "Economics", "Wirtschaftswissenschaften"),
    ("pae", "Educational Science", "Erziehungswissenschaften, Fachdidaktiken"),
    ("phi", "Philosophy", "Philosophie"),
    ("phy", "Physics", "Physik"),
    ("sow", "Social Sciences", "Sozialwissenschaften"),
    ("tec", "Engineering", "Technik allgemein"),
    ("ver", "Traffic engineering", "Verkehrstechnik, Verkehrswesen"),
    ("ber", "Mining", "Bergbau"),
    ("bio", "Life Sciences", "Biowissenschaften, Biologie"),
    ("cet", "Chemical and environmental engineering", "Chemische und Umwelttechnik"),
    ("geo", "Earth Sciences", "Geowissenschaften, Geographie"),
    ("hor", "Horticulture", "Gartenbau"),
    ("jur", "Law", "Rechtswissenschaften"),
    ("mas", "Mechanical engineering, energy technology", "Maschinenbau, Energietechnik"),
    ("med", "Medical technology", "Medizintechnik"),
    ("meda", "Medicine", "Medizin"),
    ("rel", "Study of religions", "Religionswissenschaft, Theologie"),
    ("rest", "Other subjects", "Konnte nicht zugeordnet werden"),
    ("spo", "Sports Science", "Sportwissenschaft"),
]
DOMAIN_BY_ABBR = {abbr: {"en": en, "de": de} for abbr, en, de in DOMAIN_ROWS}
DOMAIN_CODES = list(DOMAIN_BY_ABBR.keys())

# Match either "linsearch:abc" or bare "abc" (word-bounded), case-insensitive
CURIE_PATTERNS = {c: re.compile(rf"\blinsearch:{re.escape(c)}\b", re.IGNORECASE) for c in DOMAIN_CODES}
CODE_PATTERNS  = {c: re.compile(rf"\b{re.escape(c)}\b", re.IGNORECASE)            for c in DOMAIN_CODES}

def get_folder_input(prompt_text, default=None):
    if len(sys.argv) > 1 and prompt_text.startswith("Enter the path to your data"):
        return sys.argv[1]
    path = input(prompt_text).strip()
    if not path and default:
        return default
    return path

# --- Inputs ---
ROOT = get_folder_input("Enter the path to your data folder (e.g., ./library-records-dataset/data): ")
if not os.path.isdir(ROOT):
    raise ValueError(f"The provided data folder does not exist: {ROOT}")

OUTPUT_DIR = get_folder_input("Enter output directory (press Enter to use current working directory): ",
                              os.getcwd())
if not os.path.isdir(OUTPUT_DIR):
    raise ValueError(f"The provided output directory does not exist: {OUTPUT_DIR}")

# Split roots
SPLIT_DIRS = {
    "train": [os.path.join(ROOT, "train")],
    "dev":   [os.path.join(ROOT, "dev")],
    "test":  [os.path.join(ROOT, "test", "gold-standard-testset"), os.path.join(ROOT, "test")],
}

def count_domains_in_subject_value(val: str) -> dict:
    """Return {abbr: count(0/1)} for a subject string (only if it starts with DOMAIN_PREFIX)."""
    if not (isinstance(val, str) and val.startswith(DOMAIN_PREFIX)):
        return {}
    hits = {}
    for code in DOMAIN_CODES:
        if CURIE_PATTERNS[code].search(val) or CODE_PATTERNS[code].search(val):
            hits[code] = hits.get(code, 0) + 1
    # We count presence per subject value (multiple different codes in one value each add 1)
    return hits

def count_file_domains(path: str) -> dict:
    """Return total occurrences per domain in one JSON-LD file: {abbr: count}."""
    totals = defaultdict(int)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return totals

    graph = data.get("@graph", [])
    if not isinstance(graph, list):
        return totals

    for item in graph:
        subj = item.get("subject")
        if subj is None:
            continue
        if isinstance(subj, list):
            for s in subj:
                if isinstance(s, str):
                    for abbr, c in count_domains_in_subject_value(s).items():
                        totals[abbr] += c
        elif isinstance(subj, str):
            for abbr, c in count_domains_in_subject_value(subj).items():
                totals[abbr] += c
    return totals

# --- Aggregations ---
# 1) per split × domain
occ_by_split_domain = defaultdict(int)  # (split, domain) -> occurrences
# 2) per type × lang × domain (across all splits)
occ_by_type_lang_domain = defaultdict(int)  # (type, lang, domain) -> occurrences

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
                per_file = count_file_domains(fpath)

                # aggregate into both views
                for abbr, c in per_file.items():
                    occ_by_split_domain[(split, abbr)] += c
                    occ_by_type_lang_domain[(doc_type, lang, abbr)] += c

# --- Build CSV #1: per split × domain + ALL row per domain ---
rows_split = []
overall_by_domain = defaultdict(int)

for (split, abbr), total in occ_by_split_domain.items():
    info = DOMAIN_BY_ABBR[abbr]
    rows_split.append({
        "Split": split,
        "Domain Abbrev": abbr,
        "Domain (English)": info["en"],
        "Domain (German)": info["de"],
        "Total Occurrences": total
    })
    overall_by_domain[abbr] += total

# add ALL per domain
for abbr, total in overall_by_domain.items():
    info = DOMAIN_BY_ABBR[abbr]
    rows_split.append({
        "Split": "ALL",
        "Domain Abbrev": abbr,
        "Domain (English)": info["en"],
        "Domain (German)": info["de"],
        "Total Occurrences": total
    })

# ensure domains with zero counts appear for each split and ALL
for split in ["train", "dev", "test", "ALL"]:
    for abbr in DOMAIN_CODES:
        if not any(r["Split"] == split and r["Domain Abbrev"] == abbr for r in rows_split):
            info = DOMAIN_BY_ABBR[abbr]
            rows_split.append({
                "Split": split,
                "Domain Abbrev": abbr,
                "Domain (English)": info["en"],
                "Domain (German)": info["de"],
                "Total Occurrences": 0
            })

df_split = pd.DataFrame(rows_split).sort_values(["Split", "Domain Abbrev"]).reset_index(drop=True)
df_split.to_csv(os.path.join(OUTPUT_DIR, OUT_SPLIT_DOMAIN), index=False)

# --- Build CSV #2: per type × lang × domain (no ALL rows) ---
rows_type_lang = []
for (doc_type, lang, abbr), total in occ_by_type_lang_domain.items():
    info = DOMAIN_BY_ABBR[abbr]
    rows_type_lang.append({
        "Type": doc_type,
        "Lang": lang,
        "Domain Abbrev": abbr,
        "Domain (English)": info["en"],
        "Domain (German)": info["de"],
        "Total Occurrences": total
    })

# include zero rows for completeness (grid across all types/langs/domains)
for doc_type in DOC_TYPES:
    for lang in LANGS:
        for abbr in DOMAIN_CODES:
            if not any(
                r["Type"] == doc_type and r["Lang"] == lang and r["Domain Abbrev"] == abbr
                for r in rows_type_lang
            ):
                info = DOMAIN_BY_ABBR[abbr]
                rows_type_lang.append({
                    "Type": doc_type,
                    "Lang": lang,
                    "Domain Abbrev": abbr,
                    "Domain (English)": info["en"],
                    "Domain (German)": info["de"],
                    "Total Occurrences": 0
                })

df_type_lang = pd.DataFrame(rows_type_lang).sort_values(
    ["Type", "Lang", "Domain Abbrev"]
).reset_index(drop=True)
df_type_lang.to_csv(os.path.join(OUTPUT_DIR, OUT_TYPE_LANG_DOMAIN), index=False)
