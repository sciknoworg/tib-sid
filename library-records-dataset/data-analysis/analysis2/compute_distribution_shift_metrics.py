#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute KL divergence, Jensen–Shannon divergence, and Chi-Squared tests
between split-wise subject distributions for the subject-indexing dataset.

Layout assumed:
  <root>/{train,dev,test}/{Article,Book,Conference,Report,Thesis}/{en,de}/*.jsonld

What it does:
- Aggregates per-subject document-level presence counts per split (train/dev/test).
- Builds smoothed probability vectors over the union of subjects.
- Computes pairwise:
    * KL(P||Q) for each direction (nats)
    * JSD(P, Q) (bits, bounded in [0,1])
    * Chi-Squared test of independence on a 2×K contingency table (rows=splits, cols=subjects)
      - χ² statistic and degrees of freedom
      - p-value if SciPy is installed; otherwise NaN

Output:
  distribution_shift_metrics.csv  (one row per metric/value)
"""

import os
import sys
import json
from collections import defaultdict
import numpy as np
import pandas as pd

# Try to provide p-values if SciPy is available
try:
    from scipy.stats import chi2 as _scipy_chi2
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ================== CONFIG ==================
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["en", "de"]

# Smoothing for probability vectors (avoid zeros for KL)
SMOOTH_EPS = 1e-12

# Output filename
METRICS_CSV = "distribution_shift_metrics.csv"
# ============================================


def get_paths():
    """Ask for input data root and output dir (CLI arg for input is allowed)."""
    if len(sys.argv) > 1:
        root = sys.argv[1]
        print(f"Using data folder from argument: {root}")
    else:
        root = input("Enter the path to your data folder (e.g., ./library-records-dataset/data): ").strip()
    if not os.path.isdir(root):
        raise ValueError(f"Data folder does not exist: {root}")

    out_dir = input("Enter output directory (press Enter for current working directory): ").strip() or os.getcwd()
    if not os.path.isdir(out_dir):
        raise ValueError(f"Output directory does not exist: {out_dir}")
    return root, out_dir


def split_dirs(root):
    """Resolve actual split base dirs (handles test/gold-standard-testset)."""
    candidates = {
        "train": [os.path.join(root, "train")],
        "dev":   [os.path.join(root, "dev")],
        "test":  [os.path.join(root, "test", "gold-standard-testset"),
                  os.path.join(root, "test")],  # fallback
    }
    resolved = {}
    for split, options in candidates.items():
        base = next((d for d in options if os.path.isdir(d)), None)
        if base:
            resolved[split] = base
        else:
            print(f"⚠️  Skipping split '{split}' — base folder not found.")
    return resolved


def walk_files(base_dir):
    """Yield (type, lang, filepath) under base_dir for the configured DOC_TYPES/LANGS."""
    for doc_type in DOC_TYPES:
        for lang in LANGS:
            lang_dir = os.path.join(base_dir, doc_type, lang)
            if not os.path.isdir(lang_dir):
                continue
            for fname in os.listdir(lang_dir):
                if fname.endswith(".jsonld"):
                    yield doc_type, lang, os.path.join(lang_dir, fname)


def build_id_to_name_map(graph):
    """@graph → mapping @id -> sameAs (string, first if list)."""
    m = {}
    for item in graph:
        if "@id" in item and "sameAs" in item:
            same = item["sameAs"]
            if isinstance(same, str):
                m[item["@id"]] = same
            elif isinstance(same, list):
                for v in same:
                    if isinstance(v, str):
                        m[item["@id"]] = v
                        break
    return m


def extract_subject_ids(graph):
    """
    Collect all dcterms:subject @id values across items in the graph.
    Returns a set of subject IDs present in the file (deduplicated per file).
    """
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


# ---------- Divergence helpers ----------
def _prob_vector(subject_counts, subjects, count_key, eps=SMOOTH_EPS):
    """
    Build a smoothed probability vector over the UNION of subjects for a given split.
    Uses small additive smoothing eps to avoid zeros (KL undefined if Q(i)=0 with P(i)>0).
    """
    v = np.array([subject_counts[sid][count_key] for sid in subjects], dtype=float)
    v = v + eps
    s = v.sum()
    if s <= 0:
        v = np.full_like(v, fill_value=1.0 / len(v))
    else:
        v = v / s
    return v


def _choose_log(logbase="e"):
    if logbase == "e":
        return np.log
    if logbase == 2:
        return np.log2
    if logbase == 10:
        return np.log10
    raise ValueError("logbase must be 'e', 2, or 10")


def kl_divergence(P, Q, logbase="e"):
    """D_KL(P || Q) = sum_i P(i) * log(P(i)/Q(i))."""
    log = _choose_log(logbase)
    return float(np.sum(P * (log(P) - log(Q))))


def jensen_shannon_divergence(P, Q, logbase=2):
    """JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), M=0.5*(P+Q)."""
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M, logbase) + 0.5 * kl_divergence(Q, M, logbase)
# ----------------------------------------


# ---------- Chi-Squared (2×K) ----------
def chi2_two_row(counts_a: np.ndarray, counts_b: np.ndarray):
    """
    Chi-Squared test of independence for a 2×K table.

    Parameters
    ----------
    counts_a, counts_b : 1D arrays of nonnegative counts (same length K)

    Returns
    -------
    chi2_stat : float
    dof       : int   = (2-1)*(K-1) = K-1
    p_value   : float (if SciPy available), else np.nan
    """
    # Filter out columns where both are zero (no information)
    mask = (counts_a + counts_b) > 0
    A = counts_a[mask].astype(float)
    B = counts_b[mask].astype(float)

    if A.size == 0:
        return 0.0, 0, float("nan")

    # Build 2xK table
    T = np.vstack([A, B])
    total = T.sum()
    row_sums = T.sum(axis=1, keepdims=True)  # shape (2,1)
    col_sums = T.sum(axis=0, keepdims=True)  # shape (1,K)

    # Expected under independence: E = (row_sums * col_sums) / total
    E = (row_sums @ col_sums) / total

    # Avoid divide-by-zero (should not happen after masking)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_terms = (T - E) ** 2 / E
        chi2_terms = np.nan_to_num(chi2_terms, nan=0.0, posinf=0.0, neginf=0.0)

    chi2_stat = float(chi2_terms.sum())
    dof = max(0, A.size - 1)

    if _HAVE_SCIPY and dof > 0:
        # Survival function gives P(X >= chi2_stat)
        p_value = float(_scipy_chi2.sf(chi2_stat, df=dof))
    else:
        p_value = float("nan")

    return chi2_stat, dof, p_value
# ----------------------------------------


def main():
    root, out_dir = get_paths()
    bases = split_dirs(root)

    # subject_counts: subject_id -> {"train": int, "dev": int, "test": int}
    subject_counts = defaultdict(lambda: {"train": 0, "dev": 0, "test": 0})

    # Traverse and count subject presence per file per split
    for split, base_dir in bases.items():
        for _type, _lang, fpath in walk_files(base_dir):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            graph = data.get("@graph", [])
            if not isinstance(graph, list):
                continue

            subj_ids = extract_subject_ids(graph)
            if not subj_ids:
                continue

            # Count one occurrence per subject per file (presence-based)
            for sid in subj_ids:
                subject_counts[sid][split] += 1

    if not subject_counts:
        raise RuntimeError("No subjects found. Check your dataset root and structure.")

    # Universe of subjects (union)
    subject_universe = list(subject_counts.keys())

    # Probability vectors (smoothed) per split
    p_train = _prob_vector(subject_counts, subject_universe, "train", eps=SMOOTH_EPS)
    p_dev   = _prob_vector(subject_counts, subject_universe, "dev",   eps=SMOOTH_EPS)
    p_test  = _prob_vector(subject_counts, subject_universe, "test",  eps=SMOOTH_EPS)

    # Count vectors per split (for chi-square)
    c_train = np.array([subject_counts[sid]["train"] for sid in subject_universe], dtype=float)
    c_dev   = np.array([subject_counts[sid]["dev"]   for sid in subject_universe], dtype=float)
    c_test  = np.array([subject_counts[sid]["test"]  for sid in subject_universe], dtype=float)

    # KL (nats)
    KL_train_dev_e  = kl_divergence(p_train, p_dev,  logbase="e")
    KL_dev_train_e  = kl_divergence(p_dev,  p_train, logbase="e")
    KL_train_test_e = kl_divergence(p_train, p_test, logbase="e")
    KL_test_train_e = kl_divergence(p_test,  p_train, logbase="e")
    KL_dev_test_e   = kl_divergence(p_dev,   p_test,  logbase="e")
    KL_test_dev_e   = kl_divergence(p_test,  p_dev,   logbase="e")

    # JSD (bits, bounded [0,1])
    JSD_train_dev_b  = jensen_shannon_divergence(p_train, p_dev,  logbase=2)
    JSD_train_test_b = jensen_shannon_divergence(p_train, p_test, logbase=2)
    JSD_dev_test_b   = jensen_shannon_divergence(p_dev,  p_test,  logbase=2)

    # Chi-Squared (2×K)
    chi2_td, dof_td, p_td = chi2_two_row(c_train, c_dev)
    chi2_tt, dof_tt, p_tt = chi2_two_row(c_train, c_test)
    chi2_dt, dof_dt, p_dt = chi2_two_row(c_dev,   c_test)

    # Summary to CSV
    rows = [
        {"Metric": "Subject_Universe_Size", "Value": len(subject_universe)},
        {"Metric": "Smoothing_Eps",          "Value": SMOOTH_EPS},

        {"Metric": "KL_Train||Dev_nats",     "Value": round(KL_train_dev_e,  6)},
        {"Metric": "KL_Dev||Train_nats",     "Value": round(KL_dev_train_e,  6)},
        {"Metric": "KL_Train||Test_nats",    "Value": round(KL_train_test_e, 6)},
        {"Metric": "KL_Test||Train_nats",    "Value": round(KL_test_train_e, 6)},
        {"Metric": "KL_Dev||Test_nats",      "Value": round(KL_dev_test_e,   6)},
        {"Metric": "KL_Test||Dev_nats",      "Value": round(KL_test_dev_e,   6)},

        {"Metric": "JSD_Train_Dev_bits",     "Value": round(JSD_train_dev_b,  6)},
        {"Metric": "JSD_Train_Test_bits",    "Value": round(JSD_train_test_b, 6)},
        {"Metric": "JSD_Dev_Test_bits",      "Value": round(JSD_dev_test_b,   6)},

        {"Metric": "Chi2_Train_vs_Dev_stat", "Value": round(chi2_td, 6)},
        {"Metric": "Chi2_Train_vs_Dev_dof",  "Value": int(dof_td)},
        {"Metric": "Chi2_Train_vs_Dev_p",    "Value": (round(p_td, 6) if np.isfinite(p_td) else "NaN")},

        {"Metric": "Chi2_Train_vs_Test_stat","Value": round(chi2_tt, 6)},
        {"Metric": "Chi2_Train_vs_Test_dof", "Value": int(dof_tt)},
        {"Metric": "Chi2_Train_vs_Test_p",   "Value": (round(p_tt, 6) if np.isfinite(p_tt) else "NaN")},

        {"Metric": "Chi2_Dev_vs_Test_stat",  "Value": round(chi2_dt, 6)},
        {"Metric": "Chi2_Dev_vs_Test_dof",   "Value": int(dof_dt)},
        {"Metric": "Chi2_Dev_vs_Test_p",     "Value": (round(p_dt, 6) if np.isfinite(p_dt) else "NaN")},
    ]
    df = pd.DataFrame(rows)

    out_dir = out_dir  # keep name stable
    out_csv = os.path.join(out_dir, METRICS_CSV)
    df.to_csv(out_csv, index=False)

    print("\n✅ KL, JSD, and Chi-Squared computed over subject presence distributions.")
    print(f"   Subject universe size: {len(subject_universe)}")
    print(f"   Smoothing epsilon:     {SMOOTH_EPS}")
    if not _HAVE_SCIPY:
        print("   (p-values for Chi-Squared are 'NaN' — install SciPy for p-values.)")
    print(f"📄 Saved metrics to:      {out_csv}\n")


if __name__ == "__main__":
    main()
