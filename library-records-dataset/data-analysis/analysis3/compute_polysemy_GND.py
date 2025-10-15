#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect polysemy in a single GND JSON file and write:
  1) gnd_<label-source>_polysemous_labels.csv
  2) gnd_<label-source>_polysemy_summary.json

Polysemy: the SAME label string (after normalization) appears under >1 distinct "Code".
By default we consider ONLY preferred names (so unique labels ≤ number of codes).
Use --label-source all to include alternate names as well.
"""

import os
import re
import json
import argparse
import unicodedata
from collections import defaultdict
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Detect polysemy in GND (per-label output).")
    ap.add_argument("--gnd", required=True, help="Path to GND JSON file (single top-level array).")
    ap.add_argument("--out", default=".", help="Output directory (default: current dir).")
    ap.add_argument("--label-source",
                    choices=["pref", "alt", "all"],
                    default="pref",
                    help="Which labels to consider: preferred names only (pref, default), alternates (alt), or both (all).")
    ap.add_argument("--normalize",
                    choices=["none", "lower", "lower_ascii", "strict"],
                    default="lower",
                    help=("Label normalization:\n"
                          "  none        : use labels as-is\n"
                          "  lower       : Unicode NFKC + lowercase (default)\n"
                          "  lower_ascii : lower + ASCII fold (remove accents)\n"
                          "  strict      : lower + ASCII fold + strip punctuation/extra spaces"))
    ap.add_argument("--min-alt-len", type=int, default=1,
                    help="Ignore alternate names shorter than N chars (default: 1). Only used if label-source includes 'alt'.")
    return ap.parse_args()

def normalize_label(s: str, mode: str = "lower") -> str:
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", s).strip()
    if mode == "none":
        return s
    s = s.lower()
    if mode in ("lower_ascii", "strict"):
        s = "".join(c for c in unicodedata.normalize("NFKD", s)
                    if not unicodedata.combining(c))
    if mode == "strict":
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # File name prefix reflects label-source
    prefix = f"gnd_{args.label_source}_"

    with open(args.gnd, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("GND file must be a JSON array.")

    all_codes = set()
    label_map = defaultdict(lambda: defaultdict(set))  # label_norm -> code -> roles {"pref","alt"}
    label_raw_example = {}

    include_pref = args.label_source in ("pref", "all")
    include_alt  = args.label_source in ("alt", "all")

    for rec in data:
        code = rec.get("Code")
        if not code:
            continue
        all_codes.add(code)

        if include_pref:
            pref = rec.get("Name")
            if isinstance(pref, str) and pref.strip():
                pnorm = normalize_label(pref, args.normalize)
                if pnorm:
                    label_map[pnorm][code].add("pref")
                    label_raw_example.setdefault(pnorm, pref)

        if include_alt:
            alts = rec.get("Alternate Name") or []
            if not isinstance(alts, list):
                alts = [alts] if alts else []
            for a in alts:
                if not isinstance(a, str):
                    continue
                a = a.strip()
                if len(a) < args.min_alt_len:
                    continue
                anorm = normalize_label(a, args.normalize)
                if not anorm:
                    continue
                label_map[anorm][code].add("alt")
                label_raw_example.setdefault(anorm, a)

    rows = []
    for label_norm, code_roles in label_map.items():
        codes = sorted(code_roles.keys())
        if len(codes) <= 1:
            continue
        roles_per_code = [f"{c}:{'+'.join(sorted(code_roles[c]))}" for c in codes]
        raw_example = label_raw_example.get(label_norm, label_norm)
        pref_count = sum(1 for c in codes if "pref" in code_roles[c])
        alt_count  = sum(1 for c in codes if "alt"  in code_roles[c])
        rows.append({
            "LabelNorm": label_norm,
            "RawLabelExample": raw_example,
            "NumCodes": len(codes),
            "Codes": "; ".join(codes),
            "RolesPerCode": "; ".join(roles_per_code),
            "PrefCount": pref_count,
            "AltCount": alt_count
        })

    df_poly = pd.DataFrame(rows).sort_values(
        ["NumCodes", "LabelNorm"], ascending=[False, True]
    )

    out_csv = os.path.join(out_dir, f"{prefix}polysemous_labels.csv")
    df_poly.to_csv(out_csv, index=False, encoding="utf-8-sig")  # BOM for Excel

    unique_labels_considered = len(label_map)
    num_poly = int(df_poly.shape[0]) if not df_poly.empty else 0
    pct_poly = (num_poly / unique_labels_considered) if unique_labels_considered else 0.0

    summary = {
        "GND_File": os.path.abspath(args.gnd),
        "TotalCodesInFile": len(all_codes),
        "LabelSource": args.label_source,
        "Normalization": args.normalize,
        "MinAltLen": int(args.min_alt_len),
        "UniqueLabels_Considered": unique_labels_considered,
        "NumPolysemousLabels": num_poly,
        "PctPolysemousLabels": round(pct_poly, 6),
        "Example": (df_poly.iloc[0].to_dict() if num_poly else None)
    }
    out_json = os.path.join(out_dir, f"{prefix}polysemy_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved:")
    print(f"  • {out_csv}")
    print(f"  • {out_json}")
    print(f"\n📦 Codes in file:             {len(all_codes):,}")
    print(f"🧾 Unique labels considered:  {unique_labels_considered:,}  (source: {args.label_source})")
    print(f"🔁 Polysemous labels:         {num_poly:,}  ({pct_poly:.2%})")
    if num_poly:
        print("   Example:", df_poly.iloc[0]["LabelNorm"], "→", df_poly.iloc[0]["Codes"])
    print("Done.")

if __name__ == "__main__":
    main()
