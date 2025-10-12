import os
import sys
import pandas as pd

# ---------------- CONFIG ----------------
DOC_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGS = ["en", "de"]
SPLIT_DIRS = {
    "train": ["train"],
    "dev": ["dev"],
    "test": ["test/gold-standard-testset", "test"]  # handles both layouts
}
OUTPUT_FILENAME = "record_counts_by_split.csv"
# ----------------------------------------


def get_input_output_paths():
    """Ask for input data folder and optional output directory."""
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


def count_jsonld_files(folder):
    """Count number of .jsonld files inside folder (non-recursive)."""
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.endswith(".jsonld"))


def main():
    root, out_dir = get_input_output_paths()

    rows = []

    for split, paths in SPLIT_DIRS.items():
        # Resolve the first existing folder (handles test/gold-standard-testset)
        base_dir = next((os.path.join(root, p) for p in paths if os.path.isdir(os.path.join(root, p))), None)
        if not base_dir:
            print(f"⚠️  Skipping split '{split}' — not found.")
            continue

        total_records = 0
        # Count all .jsonld files across type/lang dirs
        for doc_type in DOC_TYPES:
            for lang in LANGS:
                lang_dir = os.path.join(base_dir, doc_type, lang)
                total_records += count_jsonld_files(lang_dir)

        # Now record per Type–Lang pair
        for doc_type in DOC_TYPES:
            for lang in LANGS:
                lang_dir = os.path.join(base_dir, doc_type, lang)
                cnt = count_jsonld_files(lang_dir)
                if cnt == 0:
                    continue
                rows.append({
                    "Split": split,
                    "Type": doc_type,
                    "Lang": lang,
                    "Count": cnt,
                    "Total Record": total_records
                })

    df = pd.DataFrame(rows).sort_values(["Split", "Type", "Lang"])
    output_path = os.path.join(out_dir, OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved counts to: {output_path}")


if __name__ == "__main__":
    main()
