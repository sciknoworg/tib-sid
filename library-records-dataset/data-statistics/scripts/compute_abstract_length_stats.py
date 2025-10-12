import os
import sys
import json
import pandas as pd
from statistics import mean

# ---------------- CONFIG ----------------
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["en", "de"]
SPLIT_DIRS = {
    "train": ["train"],
    "dev": ["dev"],
    "test": ["test/gold-standard-testset", "test"]
}
OUTPUT_FILENAME = "abstract_length_stats.csv"
# ----------------------------------------


def get_paths():
    """Ask user for data folder and optional output directory."""
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


def count_tokens(text: str) -> int:
    """Return number of tokens split by spaces."""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def walk_files(base_dir):
    """Yield (type, lang, filepath) under base_dir."""
    for doc_type in DOC_TYPES:
        for lang in LANGS:
            lang_dir = os.path.join(base_dir, doc_type, lang)
            if not os.path.isdir(lang_dir):
                continue
            for fname in os.listdir(lang_dir):
                if fname.endswith(".jsonld"):
                    yield doc_type, lang, os.path.join(lang_dir, fname)


def extract_abstract_lengths(path):
    """Extract abstract token length from JSONLD file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    lengths = []
    graph = data.get("@graph", [])
    if not isinstance(graph, list):
        return []

    for item in graph:
        abstract = item.get("abstract")
        if isinstance(abstract, str):
            tok_len = count_tokens(abstract)
            if tok_len > 0:
                lengths.append(tok_len)
    return lengths


def main():
    root, out_dir = get_paths()
    rows = []

    for split, paths in SPLIT_DIRS.items():
        # Resolve the first existing folder
        base_dir = next((os.path.join(root, p) for p in paths if os.path.isdir(os.path.join(root, p))), None)
        if not base_dir:
            print(f"⚠️  Skipping split '{split}' — folder not found.")
            continue

        for doc_type in DOC_TYPES:
            for lang in LANGS:
                lang_dir = os.path.join(base_dir, doc_type, lang)
                if not os.path.isdir(lang_dir):
                    continue

                lengths = []
                for fname in os.listdir(lang_dir):
                    if not fname.endswith(".jsonld"):
                        continue
                    fpath = os.path.join(lang_dir, fname)
                    file_lengths = extract_abstract_lengths(fpath)
                    lengths.extend(file_lengths)

                if lengths:
                    rows.append({
                        "Split": split,
                        "Type": doc_type,
                        "Lang": lang,
                        "Min": min(lengths),
                        "Max": max(lengths),
                        "Mean": round(mean(lengths), 1)
                    })

    df = pd.DataFrame(rows).sort_values(["Split", "Type", "Lang"])
    output_path = os.path.join(out_dir, OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")


if __name__ == "__main__":
    main()
