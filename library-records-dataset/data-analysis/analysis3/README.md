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

### 🧩 Using Embeddings for GND Subject Similarity

To run the embedding experiments on GPU on Windows, activate the appropriate virtual environment with **Python 3.11** and **torch-directml** support:

```powershell
.\dml311\Scripts\Activate.ps1
```

If using CPU (e.g., because `sentence-transformers` does not yet support DirectML on Windows), activate instead:

```powershell
.\cpu313\Scripts\Activate.ps1
```

The main script for computing pairwise similarities is:

```bash
python gnd_embed_and_polysemy.py
```

This script encodes GND subject labels either as **name-only** or **contextualized** (name + definition + alternates) embeddings, compares them using cosine similarity, and outputs per-term and summary statistics to the `./embeddings` folder.  
By default, the script supports both **Hugging Face** and **Sentence-Transformers (SBERT)** backends.

---

#### 💻 Models Tested

Initial experiments were conducted using the **Hugging Face** backend (not Sentence-Transformers), producing embeddings stored under `./embeddings/`.  
The following models were evaluated for German-language subject disambiguation:

- [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) — ✅ tested (name and name+context views complete)  
- [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) — ✅ currently running (name view complete, name+context in progress)  
- [jinaai/jina-embeddings-v2-base-de](https://huggingface.co/jinaai/jina-embeddings-v2-base-de) — ✅ tested (German bilingual specialization, suitable for polysemy analysis)

Models that required **mean-pooling fallback** performed less consistently, suggesting that **Sentence-Transformer–based embeddings** yield more stable similarity structures. However, since Sentence-Transformers currently lacks DirectML support, all such tests must be executed on **CPU**.

---

#### 🧪 Embedding Smoke Tests 

Before large-scale embedding runs, the following **smoke tests** were used to verify cosine similarity behavior between semantically related and unrelated German terms.

Example 1: `google/embeddinggemma-300m`
```python
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("google/embeddinggemma-300m", device="cpu")
print("Loaded on:", next(m.parameters()).device)
sents = ["Koordination", "Zusammenarbeit", "Kartierung"]
emb = m.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
print("cos(Koordination, Zusammenarbeit) =", float(util.cos_sim(emb[0], emb[1])))
print("cos(Koordination, Kartierung)     =", float(util.cos_sim(emb[0], emb[2])))
```
Output:
```
cos(Koordination, Zusammenarbeit) = 0.6344
cos(Koordination, Kartierung)     = 0.5043
✅ CPU smoke test looks sane (0.63 > 0.50)
```

Example 2: `jinaai/jina-embeddings-v2-base-de`
```python
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-de",
    device="cpu",
    trust_remote_code=True   # crucial for Jina
)
sents = ["Koordination", "Zusammenarbeit", "Kartierung"]
emb = m.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
print("cos(Koordination, Zusammenarbeit) =", float(util.cos_sim(emb[0], emb[1])))
print("cos(Koordination, Kartierung)     =", float(util.cos_sim(emb[0], emb[2])))
```
Output:
```
cos(Koordination, Zusammenarbeit) = 0.7756
cos(Koordination, Kartierung)     = 0.4116
✅ CPU smoke test looks sane (0.78 > 0.41)
```

Example 3: `Qwen/Qwen3-Embedding-8B`
```python
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cpu")
sents = ["Koordination", "Zusammenarbeit", "Kartierung"]
emb = m.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
print("cos(Koordination, Zusammenarbeit) =", float(util.cos_sim(emb[0], emb[1])))
print("cos(Koordination, Kartierung)     =", float(util.cos_sim(emb[0], emb[2])))
```
Output:
```
cos(Koordination, Zusammenarbeit) = 0.8717
cos(Koordination, Kartierung)     = 0.6895
✅ CPU smoke test looks sane (0.87 > 0.69)
```

---

## 🧠 Summary

Use the smoke tests above to select a model that provides the clearest semantic separation between related and unrelated German terms.  
Once chosen, run the full-scale polysemy analysis with `gnd_embed_and_polysemy.py`, adjusting backend (`HF` or `SBERT`), similarity threshold (e.g., `0.9`), and view type (`name` or `context`) interactively at runtime.

Other alternative for German

https://huggingface.co/LSX-UniWue/ModernGBERT_1B
