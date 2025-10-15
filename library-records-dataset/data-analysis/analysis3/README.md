# Assessing Polysemy

This folder contains two scripts to quantify “same-name” cases and assess whether they reflect true polysemy or cataloging reuse.

## 1) Library Records

Analyzes dataset records to find subject labels linked to multiple GND IDs.

Run:
```bash
python compute_polysemy_library_records.py --data ../../data --out ./library-records
```

**Outputs (in `library-records/`):**
- `polysemy_label_id_breakdown.csv` – one row per (label, ID)
- `polysemy_by_label.csv` – one row per label with entropy and dominance measures
- `polysemy_summary.json` – summary stats and top examples

## 2) GND Taxonomy

Checks the GND JSON for labels mapping to multiple codes.

Run with preferred names:
```bash
python compute_polysemy_GND.py --gnd ./GND_subjects.json --out ./GND
```

Include alternates:
```bash
python compute_polysemy_GND.py --gnd ./GND_subjects.json --out ./GND --label-source all
```

**Outputs (in `GND/`):**
- `gnd_<label-source>_polysemous_labels.csv` – one row per label mapping to >1 code
- `gnd_<label-source>_polysemy_summary.json` – corpus-level summary

Both scripts output UTF-8 CSVs. Use `--help` for more options.
