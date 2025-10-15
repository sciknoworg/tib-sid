#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count non-empty values per property in a GND subjects JSON file.

Input JSON format: a list of dicts, each representing one GND subject record.
Each record may contain the following properties (strings or arrays of strings):
  - "Code"                    (string)
  - "Classification Number"   (string)
  - "Classification Name"     (string)
  - "Name"                    (string)
  - "Alternate Name"          (array of strings)
  - "Related Subjects"        (array of strings)
  - "Source"                  (string)
  - "Source URL"              (string, optional)
  - "Definition"              (string, optional)

Usage:
  python count_property_coverage.py --input GND-subjects.json --out-csv coverage.csv
If --input is omitted, you’ll be prompted.
"""

import json
import argparse
import sys
from collections import OrderedDict
from typing import Any, List

# Schema properties and their expected value kinds
PROPS = OrderedDict([
    ("Code", "string"),
    ("Classification Number", "string"),
    ("Classification Name", "string"),
    ("Name", "string"),
    ("Alternate Name", "array"),
    ("Related Subjects", "array"),
    ("Source", "string"),
    ("Source URL", "string"),
    ("Definition", "string"),
])

def is_nonempty_string(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""

def is_nonempty_array_of_strings(v: Any) -> bool:
    if not isinstance(v, list):
        return False
    # Count as non-empty if there is at least one non-empty string item
    return any(is_nonempty_string(x) for x in v)

def has_value(prop_kind: str, v: Any) -> bool:
    if v is None:
        return False
    if prop_kind == "string":
        return is_nonempty_string(v)
    if prop_kind == "array":
        return is_nonempty_array_of_strings(v)
    # Fallback: treat truthy as present
    return bool(v)

def main():
    parser = argparse.ArgumentParser(description="Count property coverage in GND subjects JSON.")
    parser.add_argument("--input", "-i", help="Path to GND JSON file (list of records).")
    parser.add_argument("--out-csv", "-o", help="Optional path to write CSV summary.")
    args = parser.parse_args()

    input_path = args.input or input("INPUT_FILE = ").strip()
    if not input_path:
        print("❌ No input file provided.", file=sys.stderr)
        sys.exit(1)

    # Load JSON (ensure UTF-8)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("❌ Expected a JSON array of records.", file=sys.stderr)
        sys.exit(1)

    total = len(data)
    counts = {k: 0 for k in PROPS.keys()}

    for rec in data:
        if not isinstance(rec, dict):
            continue
        for prop, kind in PROPS.items():
            if prop in rec and has_value(kind, rec.get(prop)):
                counts[prop] += 1

    # Print summary
    print(f"Total records: {total}")
    print("Property coverage (non-empty values):")
    print("{:<26} {:>8} {:>10}".format("Property", "Count", "Percent"))
    print("-" * 46)
    for prop in PROPS.keys():
        cnt = counts[prop]
        pct = (cnt / total * 100) if total else 0.0
        print("{:<26} {:>8} {:>9.2f}%".format(prop, cnt, pct))

    # Optional CSV
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["property", "count_nonempty", "percent", "total_records"])
            for prop in PROPS.keys():
                cnt = counts[prop]
                pct = (cnt / total * 100) if total else 0.0
                w.writerow([prop, cnt, f"{pct:.4f}", total])
        print(f"\n📝 Wrote CSV summary to: {args.out_csv}")

if __name__ == "__main__":
    main()
