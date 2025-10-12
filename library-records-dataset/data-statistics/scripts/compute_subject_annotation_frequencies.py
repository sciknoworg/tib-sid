import os
import sys
import json
import pandas as pd
from statistics import mean

# --------- CONFIG ----------
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["en", "de"]
CSV_NAME = "subject_annotation_frequencies.csv"
# --------------------------


def get_paths():
    """Get input data folder (CLI arg or prompt) and optional output directory (defaults to input folder)."""
    if len(sys.argv) > 1:
        root = sys.argv[1]
        print(f"Using data folder from argument: {root}")
    else:
        root = input("Enter the path to your data folder (e.g., ./library-records-dataset/data): ").strip()
    if not os.path.isdir(root):
        raise ValueError(f"The provided path does not exist or is not a directory: {root}")

    out_dir = input("Enter output directory (press Enter to save next to the data folder): ").strip() or root
    if not os.path.isdir(out_dir):
        raise ValueError(f"The provided output directory does not exist: {out_dir}")
    return root, out_dir


def count_subjects_in_file(path):
    """Return total number of dcterms:subject entries across all @graph items (treat singletons as 1)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None  # unreadable -> skip

    total = 0
    graph = data.get("@graph", [])
    if not isinstance(graph, list):
        return 0

    for item in graph:
        subj = item.get("dcterms:subject")
        if subj is None:
            continue
        if isinstance(subj, list):
            total += len(subj)
        else:
            total += 1
    return total


def main():
    ROOT, OUT_DIR = get_paths()

    # Map the logical split name -> list of physical directories to search
    SPLIT_DIRS = {
        "train": [os.path.join(ROOT, "train")],
        "dev":   [os.path.join(ROOT, "dev")],
        # Test is inside 'test/gold-standard-testset'; accept 'test' fallback just in case
        "test":  [os.path.join(ROOT, "test", "gold-standard-testset"),
                  os.path.join(ROOT, "test")],
    }

    rows = []

    for split, base_dirs in SPLIT_DIRS.items():
        # choose first existing base dir (allows both test layouts)
        base_dir = next((d for d in base_dirs if os.path.isdir(d)), None)
        if not base_dir:
            continue  # split dir not present

        for doc_type in DOC_TYPES:
            for lang in LANGS:
                lang_dir = os.path.join(base_dir, doc_type, lang)
                if not os.path.isdir(lang_dir):
                    continue

                counts = []
                max_file = None
                max_count = -1

                for fname in os.listdir(lang_dir):
                    if not fname.endswith(".jsonld"):
                        continue
                    fpath = os.path.join(lang_dir, fname)
                    cnt = count_subjects_in_file(fpath)
                    if cnt is None:
                        continue  # unreadable, skip
                    counts.append(cnt)
                    if cnt > max_count:
                        max_count = cnt
                        max_file = fname

                if counts:
                    rows.append({
                        "Split": split,
                        "Type": doc_type,
                        "Lang": lang,
                        "Min": min(counts),
                        "Max": max(counts),
                        "Mean": round(mean(counts), 1),
                        "MaxFile": max_file
                    })

    df = pd.DataFrame(rows).sort_values(["Split", "Type", "Lang"]).reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, CSV_NAME)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
