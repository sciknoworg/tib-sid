#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polysemy measurement for GND subject labels:
- Traverses <data>/{train,dev,test}/{Article,Book,Conference,Report,Thesis}/{en,de}/*.jsonld
- Reads subject IDs per record (deduped per file), maps IDs to label strings
- Aggregates counts per (label, ID) and computes polysemy measures per label:
    * num_ids: number of distinct GND IDs sharing the same label string
    * total_count: total doc-level presences across all IDs for that label
    * entropy_bits: H(label) over IDs in bits
    * normalized_entropy: H / log2(num_ids)  (0..1; 1 = evenly split across IDs)
    * hhi: sum p_i^2 (concentration; 1/num_ids when even, →1 when a single ID dominates)
    * dominant_id, dominant_count, dominant_share

Outputs:
  - polysemy_label_id_breakdown.csv
  - polysemy_by_label.csv
  - polysemy_summary.json
"""

import os
import json
import math
import argparse
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# ---------------- CONFIG DEFAULTS ----------------
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["en", "de"]
LOG_EVERY = 2000
# -------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Quantify polysemy of GND subject labels.")
    ap.add_argument("--data", required=True,
                    help="Input dataset directory (e.g., ./library-records-dataset/data)")
    ap.add_argument("--out", default=".",
                    help="Output directory (default: current dir)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: maximum number of files to parse (for quick runs)")
    ap.add_argument("--normalize", choices=["none", "lower"], default="lower",
                    help="Label normalization: 'lower' (default) or 'none'")
    ap.add_argument("--min-total", type=int, default=1,
                    help="Only include (label,ID) rows with TotalCount >= this value (default 1)")
    return ap.parse_args()


def iter_files(base_dir):
    """Yield jsonld file paths under the expected DOC_TYPES/LANGS."""
    for doc_type in DOC_TYPES:
        for lang in LANGS:
            lang_dir = os.path.join(base_dir, doc_type, lang)
            if not os.path.isdir(lang_dir):
                continue
            try:
                with os.scandir(lang_dir) as it:
                    for entry in it:
                        if entry.is_file() and entry.name.endswith(".jsonld"):
                            yield entry.path
            except PermissionError:
                continue


def split_dirs(data):
    """Resolve split dirs (handles gold-standard-testset fallback)."""
    candidates = {
        "train": [os.path.join(data, "train")],
        "dev":   [os.path.join(data, "dev")],
        "test":  [os.path.join(data, "test", "gold-standard-testset"),
                  os.path.join(data, "test")],
    }
    resolved = {}
    for split, options in candidates.items():
        base = next((d for d in options if os.path.isdir(d)), None)
        if base:
            resolved[split] = base
        else:
            print(f"⚠️  Skipping split '{split}' — base folder not found.")
    return resolved


def _as_string_label(val):
    """
    Try to extract a label string from various JSON-LD patterns:
      - string directly
      - {"@value": "..."} (literal)
      - list[...] -> pick first string or {'@value': str}
    """
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        v = val.get("@value")
        if isinstance(v, str):
            return v
    if isinstance(val, list):
        for x in val:
            s = _as_string_label(x)
            if isinstance(s, str):
                return s
    return None


def build_id_to_label_map(graph):
    """
    Construct mapping: subject @id -> label string (best-effort).
    Tries common keys in priority order; falls back to 'sameAs' if used as label in your data.
    """
    KEYS = [
        "skos:prefLabel", "prefLabel", "rdfs:label", "label", "name",
        "gndo:preferredNameForTheSubjectHeading", "sameAs"
    ]
    m = {}
    for item in graph:
        sid = item.get("@id")
        if not sid:
            continue
        for k in KEYS:
            if k in item:
                lab = _as_string_label(item[k])
                if isinstance(lab, str) and lab.strip():
                    m[sid] = lab.strip()
                    break
    return m


def extract_subject_ids(graph):
    """Collect deduped dcterms:subject @id values present in the record."""
    ids = set()
    for item in graph:
        subj = item.get("dcterms:subject")
        if subj is None:
            continue
        if isinstance(subj, list):
            for s in subj:
                if isinstance(s, dict) and "@id" in s:
                    ids.add(s["@id"])
        elif isinstance(subj, dict) and "@id" in subj:
            ids.add(subj["@id"])
    return ids


def main():
    args = parse_args()
    data = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    bases = split_dirs(data)
    if not bases:
        raise RuntimeError("No splits found under --data.")

    # counts[(label_norm, subject_id)] -> dict with per-split counts + raw label sample(s)
    counts = defaultdict(lambda: {"train": 0, "dev": 0, "test": 0, "raw_label": None})
    files_parsed = 0

    # Traverse
    for split, base_dir in bases.items():
        for fpath in iter_files(base_dir):
            files_parsed += 1
            if args.limit and files_parsed > args.limit:
                print(f"⏭️  Reached file limit {args.limit}. Stopping early.")
                break
            if LOG_EVERY and files_parsed % LOG_EVERY == 0:
                print(f"… parsed {files_parsed} files (last: {os.path.basename(fpath)})")

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            graph = data.get("@graph", [])
            if not isinstance(graph, list):
                continue

            id2label = build_id_to_label_map(graph)
            subj_ids = extract_subject_ids(graph)
            if not subj_ids:
                continue

            # Document-level presence: 1 per (subject_id) per file
            for sid in subj_ids:
                raw_label = id2label.get(sid, None) or sid  # fallback to sid if label missing
                label_norm = raw_label
                if args.normalize == "lower" and isinstance(raw_label, str):
                    label_norm = raw_label.lower().strip()

                key = (label_norm, sid)
                counts[key][split] += 1
                # Keep a representative raw label string (first seen non-ID string)
                if counts[key]["raw_label"] is None and raw_label != sid:
                    counts[key]["raw_label"] = raw_label

    if not counts:
        raise RuntimeError("No (label,ID) pairs collected. Check dataset structure.")

    # ---------- Build raw breakdown table ----------
    rows = []
    for (label_norm, sid), rec in counts.items():
        tr, dv, ts = rec["train"], rec["dev"], rec["test"]
        total = tr + dv + ts
        if total < args.min_total:
            continue
        rows.append({
            "LabelNorm": label_norm,
            "RawLabelExample": rec["raw_label"] if rec["raw_label"] else (label_norm if isinstance(label_norm, str) else str(label_norm)),
            "SubjectID": sid,
            "TrainCount": tr,
            "DevCount": dv,
            "TestCount": ts,
            "TotalCount": total
        })
    breakdown_df = pd.DataFrame(rows)
    if breakdown_df.empty:
        raise RuntimeError(f"No rows after filtering with --min-total={args.min_total}.")

    breakdown_df.sort_values(["LabelNorm", "TotalCount"], ascending=[True, False], inplace=True)
    breakdown_csv = os.path.join(out_dir, "polysemy_label_id_breakdown.csv")
    breakdown_df.to_csv(breakdown_csv, index=False)

    # ---------- Aggregate to label-level polysemy ----------
    poly_rows = []
    for label_norm, group in breakdown_df.groupby("LabelNorm"):
        # per-ID totals
        id_counts = group[["SubjectID", "TotalCount"]].groupby("SubjectID")["TotalCount"].sum().sort_values(ascending=False)
        total = int(id_counts.sum())
        num_ids = int(len(id_counts))

        # proportions across IDs
        p = id_counts.values.astype(float) / total if total > 0 else np.array([1.0 / num_ids] * num_ids)
        # entropy in bits
        entropy_bits = float(-(p * np.log2(p + 1e-300)).sum()) if total > 0 else 0.0  # safeguard tiny
        normalized_entropy = float(entropy_bits / math.log2(num_ids)) if num_ids > 1 else 0.0
        hhi = float((p ** 2).sum())  # concentration index (1/num_ids when even; →1 when single ID dominates)

        dominant_id = id_counts.index[0]
        dominant_count = int(id_counts.iloc[0])
        dominant_share = float(dominant_count / total) if total > 0 else 0.0

        # Choose a representative raw label example for this normalized label
        raw_label_example = group.sort_values("TotalCount", ascending=False)["RawLabelExample"].iloc[0]

        poly_rows.append({
            "LabelNorm": label_norm,
            "RawLabelExample": raw_label_example,
            "NumDistinctIDs": num_ids,
            "TotalCount": total,
            "Entropy_bits": round(entropy_bits, 6),
            "NormalizedEntropy": round(normalized_entropy, 6),
            "HHI": round(hhi, 6),
            "DominantID": dominant_id,
            "DominantCount": dominant_count,
            "DominantShare": round(dominant_share, 6)
        })
    poly_df = pd.DataFrame(poly_rows).sort_values(
        ["NumDistinctIDs", "NormalizedEntropy", "TotalCount"],
        ascending=[False, False, False]
    )
    poly_csv = os.path.join(out_dir, "polysemy_by_label.csv")
    poly_df.to_csv(poly_csv, index=False)

    # ---------- Summary ----------
    num_labels = int(poly_df.shape[0])
    poly_mask = poly_df["NumDistinctIDs"] > 1
    num_polysemous = int(poly_mask.sum())
    pct_polysemous = float(num_polysemous / num_labels) if num_labels else 0.0
    avg_ids_poly = float(poly_df.loc[poly_mask, "NumDistinctIDs"].mean()) if num_polysemous else 0.0
    med_ids_poly = float(poly_df.loc[poly_mask, "NumDistinctIDs"].median()) if num_polysemous else 0.0

    # top ambiguous labels by normalized entropy then by NumDistinctIDs
    top_ambiguous = poly_df[poly_mask].head(20).to_dict(orient="records")

    summary = {
        "FilesParsed": int(files_parsed),
        "LabelNormalization": args.normalize,
        "MinTotalFilter": int(args.min_total),
        "NumLabels": num_labels,
        "NumPolysemousLabels": num_polysemous,
        "PctPolysemousLabels": round(pct_polysemous, 6),
        "AvgIDsPerPolysemousLabel": round(avg_ids_poly, 4),
        "MedianIDsPerPolysemousLabel": round(med_ids_poly, 4),
        "ExamplesTopAmbiguous": top_ambiguous
    }
    import json as _json
    with open(os.path.join(out_dir, "polysemy_summary.json"), "w", encoding="utf-8") as f:
        _json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved:")
    print(f"  • {breakdown_csv}")
    print(f"  • {poly_csv}")
    print(f"  • {os.path.join(out_dir, 'polysemy_summary.json')}")
    print(f"\n📊 Labels: {num_labels} | Polysemous: {num_polysemous} ({pct_polysemous:.2%})")
    if num_polysemous:
        print(f"   Avg IDs/Polysemous Label: {avg_ids_poly:.2f} | Median: {med_ids_poly:.2f}")
    print("Done.")
    

if __name__ == "__main__":
    main()
