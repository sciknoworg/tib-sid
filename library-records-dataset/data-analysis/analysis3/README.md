

Run with preferred names only (the default):
```
python detect_gnd_polysemy_min.py --gnd ./GND_subjects.json --out ./out
```

If you later want to include alternates (which will push the “unique labels” count above the number of codes), use:
```
python detect_gnd_polysemy_min.py --gnd ./GND_subjects.json --out ./out --label-source all
```

Output:

`gnd_polysemous_labels.csv`: one row per label (normalized) that maps to >1 Code, with counts and per-code details
`gnd_polysemy_summary.json`: corpus-level summary (how many labels are polysemous, examples)