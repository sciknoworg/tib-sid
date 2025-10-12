import os
import sys
import json
import pandas as pd
from collections import defaultdict

# ================== CONFIG ==================
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["en", "de"]

# Outlier params (tweak as needed)
RARITY_THRESHOLD = 1        # total count <= 1 => rarity outlier
DOMINANCE_THRESHOLD = 0.90  # >= 90% of occurrences in one split
MIN_SUPPORT = 5             # only consider dominance outliers with total >= 5

# Output filenames (CSV)
SUBJECTS_CSV = "subjects_by_split.csv"
OVERLAP_CSV = "split_overlap_summary.csv"
OUTLIERS_CSV = "outliers_by_subject.csv"
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
    """Return actual split base dirs (handles test/gold-standard-testset)."""
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


def build_id_to_name_map(graph):
    """
    From @graph, build a mapping @id -> sameAs (string).
    Only keep entries where sameAs is a string; if it's a list, take the first string.
    """
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
    Returns a set of subject IDs present in the file.
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


def main():
    root, out_dir = get_paths()
    bases = split_dirs(root)

    # subject_counts: subject_id -> {"name": str, "train": int, "dev": 0, "test": 0}
    subject_counts = defaultdict(lambda: {"name": None, "train": 0, "dev": 0, "test": 0})

    # Track per-split sets (binary presence) for Jaccard
    split_sets = {"train": set(), "dev": set(), "test": set()}

    # NEW: count total .jsonld files per split (regardless of parse success)
    split_file_counts = {"train": 0, "dev": 0, "test": 0}

    for split, base_dir in bases.items():
        for _type, _lang, fpath in walk_files(base_dir):
            # Count the file for this split
            split_file_counts[split] += 1

            # Parse and aggregate subjects
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                # even if unreadable, it still counted toward file totals
                continue

            graph = data.get("@graph", [])
            if not isinstance(graph, list):
                continue

            id2name = build_id_to_name_map(graph)
            subj_ids = extract_subject_ids(graph)
            if not subj_ids:
                continue

            # Presence count per file (deduplicated within file)
            for sid in subj_ids:
                name = id2name.get(sid, sid)
                entry = subject_counts[sid]
                if entry["name"] is None:
                    entry["name"] = name
                entry[split] += 1

            # Binary presence for Jaccard
            split_sets[split].update(subj_ids)

    # -------- Build subjects_by_split.csv --------
    rows = []
    for sid, rec in subject_counts.items():
        train_c = rec["train"]
        dev_c = rec["dev"]
        test_c = rec["test"]
        total = train_c + dev_c + test_c
        present_in = sum(1 for v in (train_c, dev_c, test_c) if v > 0)
        rows.append({
            "SubjectID": sid,
            "SubjectName": rec["name"],
            "TrainCount": train_c,
            "DevCount": dev_c,
            "TestCount": test_c,
            "TotalCount": total,
            "PresentInSplits": present_in
        })
    subjects_df = pd.DataFrame(rows).sort_values(
        ["TotalCount", "PresentInSplits", "SubjectName"],
        ascending=[False, False, True]
    )
    subjects_csv_path = os.path.join(out_dir, SUBJECTS_CSV)
    subjects_df.to_csv(subjects_csv_path, index=False)

    # -------- Overlap metrics (binary Jaccard + weighted Jaccard) --------
    train_set = split_sets.get("train", set())
    dev_set   = split_sets.get("dev", set())
    test_set  = split_sets.get("test", set())

    def jaccard(a, b):
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b) if (a or b) else 0.0

    # weighted jaccard (tanimoto) over nonnegative count vectors
    def weighted_jaccard(count_key_x: str, count_key_y: str) -> float:
        min_sum = 0
        max_sum = 0
        for rec in subject_counts.values():
            x = rec[count_key_x]
            y = rec[count_key_y]
            min_sum += min(x, y)
            max_sum += max(x, y)
        if max_sum == 0:
            return 1.0  # both zero everywhere
        return round(min_sum / max_sum, 4)

    overlap_rows = [
        # NEW: file counts per split
        {"Metric": "Train_FileCount", "Value": split_file_counts.get("train", 0)},
        {"Metric": "Dev_FileCount", "Value": split_file_counts.get("dev", 0)},
        {"Metric": "Test_FileCount", "Value": split_file_counts.get("test", 0)},

        # Subject set sizes
        {"Metric": "Train_Size", "Value": len(train_set)},
        {"Metric": "Dev_Size", "Value": len(dev_set)},
        {"Metric": "Test_Size", "Value": len(test_set)},

        # Intersections & uniques
        {"Metric": "Train_Dev_Intersection", "Value": len(train_set & dev_set)},
        {"Metric": "Train_Test_Intersection", "Value": len(train_set & test_set)},
        {"Metric": "Dev_Test_Intersection", "Value": len(dev_set & test_set)},
        {"Metric": "All_Three_Intersection", "Value": len(train_set & dev_set & test_set)},
        {"Metric": "Train_Only", "Value": len(train_set - (dev_set | test_set))},
        {"Metric": "Dev_Only", "Value": len(dev_set - (train_set | test_set))},
        {"Metric": "Test_Only", "Value": len(test_set - (train_set | dev_set))},

        # Binary Jaccard
        {"Metric": "Jaccard_Train_Dev", "Value": round(jaccard(train_set, dev_set), 4)},
        {"Metric": "Jaccard_Train_Test", "Value": round(jaccard(train_set, test_set), 4)},
        {"Metric": "Jaccard_Dev_Test", "Value": round(jaccard(dev_set, test_set), 4)},

        # Weighted Jaccard (Tanimoto) using per-subject presence counts
        {"Metric": "WeightedJaccard_Train_Dev", "Value": weighted_jaccard("train", "dev")},
        {"Metric": "WeightedJaccard_Train_Test", "Value": weighted_jaccard("train", "test")},
        {"Metric": "WeightedJaccard_Dev_Test", "Value": weighted_jaccard("dev", "test")},
    ]
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_csv_path = os.path.join(out_dir, OVERLAP_CSV)
    overlap_df.to_csv(overlap_csv_path, index=False)

    # -------- Outliers --------
    outlier_rows = []
    for sid, rec in subject_counts.items():
        tr, dv, ts = rec["train"], rec["dev"], rec["test"]
        total = tr + dv + ts
        if total == 0:
            continue

        shares = {
            "train": tr / total,
            "dev":   dv / total,
            "test":  ts / total
        }
        dominant_split = max(shares, key=shares.get)
        dominant_share = shares[dominant_split]

        rarity_flag = total <= RARITY_THRESHOLD
        dominance_flag = (total >= MIN_SUPPORT) and (dominant_share >= DOMINANCE_THRESHOLD)

        outlier_rows.append({
            "SubjectID": sid,
            "SubjectName": rec["name"],
            "TrainCount": tr,
            "DevCount": dv,
            "TestCount": ts,
            "TotalCount": total,
            "DominantSplit": dominant_split,
            "DominantShare": round(dominant_share, 4),
            "IsRarityOutlier": rarity_flag,
            "IsDominanceOutlier": dominance_flag
        })

    outliers_df = pd.DataFrame(outlier_rows).sort_values(
        ["IsDominanceOutlier", "IsRarityOutlier", "TotalCount"],
        ascending=[False, False, False]
    )
    outliers_csv_path = os.path.join(out_dir, OUTLIERS_CSV)
    outliers_df.to_csv(outliers_csv_path, index=False)

    print(f"\n✅ Saved: {subjects_csv_path}")
    print(f"✅ Saved: {overlap_csv_path}")
    print(f"✅ Saved: {outliers_csv_path}")
    print(f"📦 File counts — train: {split_file_counts.get('train',0)}, dev: {split_file_counts.get('dev',0)}, test: {split_file_counts.get('test',0)}")


if __name__ == "__main__":
    main()
