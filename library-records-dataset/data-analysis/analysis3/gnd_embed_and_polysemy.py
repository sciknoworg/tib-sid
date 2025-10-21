#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise similarity counts for GND subjects with user prompts.

Two settings:
  1) name-only
  2) context = name + (definition / alternate names where present)

Features:
  - Skip acronym-like labels (not encoded, not compared).
  - User-chosen similarity threshold (e.g., 0.9 or 90%).
  - Backend: HuggingFace (HF) or Sentence-Transformers (SBERT).
  - Device: CUDA, DirectML (Windows), or CPU.
  - ANN search for scale: Annoy or PyNNDescent if installed; else chunked brute fallback.
  - Outputs:
      similar_counts__<slug>__<view>__thXX.csv
      similar_summary__<slug>__<view>__thXX.csv
  - Caches embeddings: <slug>__name_embeddings.npy / <slug>__context_embeddings.npy
"""

import os, re, json, math, time, gc, sys
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

# Optional libs
try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except Exception:
    SBERT_OK = False

try:
    from annoy import AnnoyIndex
    ANNOY_OK = True
except Exception:
    ANNOY_OK = False

try:
    import pynndescent
    PYNN_OK = True
except Exception:
    PYNN_OK = False

try:
    from tqdm import tqdm
    TQDM_OK = True
except Exception:
    TQDM_OK = False


# ------------------------- helpers -------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.sqrt(np.maximum((x * x).sum(axis=axis, keepdims=True), eps))
    return x / denom

def build_context_text(entry: Dict) -> str:
    """Name + DEF/ALT if available (compact)."""
    name = entry.get("Name", "") or ""
    definition = entry.get("Definition", "") or ""
    alts = entry.get("Alternate Name", []) or []
    if not isinstance(alts, list):
        alts = [str(alts)]
    alt_part = ", ".join([str(a) for a in alts[:12]])

    parts = [name]
    if definition:
        parts.append(f"DEF: {definition}")
    if alt_part:
        parts.append(f"ALT: {alt_part}")
    return " [SEP] ".join([p for p in parts if p])

def is_acronymish(label: str) -> bool:
    """
    Heuristic: skip all-uppercase / code-like short labels (e.g., 'MAP', 'EC 2.7.1.11', 'RPC').
    - If there is any lowercase letter, we keep it.
    - Otherwise, if (letters+digits) length in [2..15] and only A-Z0-9 plus separators, treat as acronymish.
    """
    if not label or not label.strip():
        return False
    s = label.strip()
    if re.search(r"[a-z]", s):
        return False  # has lowercase, likely a regular term
    core = re.sub(r"[^\w]", "", s, flags=re.UNICODE)  # strip spaces/punct
    if len(core) < 2:
        return False
    if len(core) <= 15 and re.fullmatch(r"[A-Z0-9._/\-() ]+", s):
        return True
    # Also if all alphabetic chars (if any) are uppercase
    letters = [ch for ch in core if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters) and len(core) <= 20:
        return True
    return False

def pick_device_interactive() -> Tuple[str, torch.device]:
    print("\n=== Device selection ===")
    print("Use GPU? Options:")
    print("  [1] CUDA (NVIDIA)")
    print("  [2] DirectML (Windows: Intel/AMD/NVIDIA)")
    print("  [3] CPU")
    choice = input("Choose 1/2/3 (default 2 for DirectML on Windows): ").strip() or "2"
    if choice == "1" and torch.cuda.is_available():
        print("-> Using CUDA")
        return "cuda", torch.device("cuda")
    if choice == "2":
        try:
            import torch_directml
            print("-> Using DirectML")
            return "dml", torch_directml.device()
        except Exception:
            print("DirectML not available, falling back to CPU.")
    print("-> Using CPU")
    return "cpu", torch.device("cpu")

def parse_threshold(th: str) -> float:
    s = (th or "").strip().replace("%","")
    if not s:
        return 0.90
    v = float(s)
    return v/100.0 if v > 1.0 else v

def ask(prompt: str, default: str = "") -> str:
    v = input(f"{prompt} [{default}]: ").strip()
    return v or default

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_subjects(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for obj in data:
        code = obj.get("Code") or obj.get("code") or ""
        name = obj.get("Name") or obj.get("name") or ""
        if code and name:
            out.append(obj)
    if not out:
        raise ValueError("No subjects with both 'Code' and 'Name'.")
    return out

# -------------- encoding backends --------------

@torch.no_grad()
def encode_hf(model_name: str, texts, batch_size, max_length, device_kind, device_obj) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    import time, gc

    print(f"\n[ENC] Loading tokenizer/model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device_obj)
    print(f"[ENC] Model ready. class={type(model).__name__}")

    total = len(texts)
    arr = None
    write_pos = 0

    ranges = [range(i, min(i + batch_size, total)) for i in range(0, total, batch_size)]
    iterator = tqdm(ranges, total=len(ranges), unit="batch", dynamic_ncols=True,
                    leave=False, desc="Encoding") if TQDM_OK else ranges

    used_sent_batches = 0
    used_fallback_batches = 0
    t0 = time.perf_counter()

    for b_ix, br in enumerate(iterator, 1):
        batch = [texts[i] for i in br]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        for k in enc:
            enc[k] = enc[k].to(device_obj)

        out = model(**enc)

        # ---- KEY CHECK: prefer model-provided sentence embeddings
        pooled = getattr(out, "sentence_embeddings", None)
        if pooled is not None:
            if used_sent_batches == 0:
                print("[ENC] Using model-provided sentence_embeddings (first seen this batch).")
            used_sent_batches += 1
        else:
            # fallback to masked mean pooling
            last_hidden = out.last_hidden_state
            attn = enc["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * attn).sum(1) / attn.sum(1).clamp(min=1)
            if used_fallback_batches == 0:
                print("[ENC] No sentence_embeddings in outputs; using mean-pool fallback.")
            used_fallback_batches += 1
        # ----

        pooled_np = pooled.detach().cpu().float().numpy()
        if arr is None:
            H = pooled_np.shape[1]
            print(f"[ENC] Embedding dim = {H}")
            arr = np.empty((total, H), dtype="float32")

        end = write_pos + pooled_np.shape[0]
        arr[write_pos:end] = pooled_np
        write_pos = end

        del enc, out, pooled, pooled_np
        if device_kind == "cuda":
            torch.cuda.empty_cache()

    dt = time.perf_counter() - t0
    print(f"[ENC] Done in {dt:.1f}s. Batches with sentence_embeddings: {used_sent_batches} | "
          f"fallback mean-pool: {used_fallback_batches}")

    del model, tok
    gc.collect()
    if device_kind == "cuda":
        torch.cuda.empty_cache()
        try: torch.cuda.ipc_collect()
        except Exception: pass

    return l2_normalize(arr, axis=1)

@torch.no_grad()
def encode_sbert(model_name: str, texts: List[str], batch_size: int, device_kind: str) -> np.ndarray:
    if not SBERT_OK:
        raise RuntimeError("sentence-transformers is not installed. Install: pip install sentence-transformers")
    dev = "cuda" if (device_kind == "cuda") else "cpu"
    model = SentenceTransformer(model_name, device=dev, trust_remote_code=True)  # <-- add this
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=TQDM_OK,
        normalize_embeddings=True,
    )
    return emb.astype("float32")


# -------------- ANN backends --------------

def topk_annoy(emb: np.ndarray, topk: int, n_trees: int = 50):
    N, D = emb.shape
    idx = AnnoyIndex(D, metric="angular")
    for i in range(N):
        idx.add_item(i, emb[i].tolist())
    idx.build(n_trees)
    def query(i):
        neigh, dists = idx.get_nns_by_item(i, topk + 1, include_distances=True)
        out = []
        for j in neigh:
            if j == i: 
                continue
            sim = float(np.dot(emb[i], emb[j]))  # re-check cosine
            out.append((j, sim))
        return out
    return query, idx

def topk_pynnd(emb: np.ndarray, topk: int):
    index = pynndescent.NNDescent(emb, n_neighbors=min(topk+1, max(5, topk+1)), metric="cosine", random_state=42, verbose=False)
    index.prepare()
    def query(i):
        nbrs, dists = index.query(emb[i:i+1], k=topk+1)
        nbrs = nbrs[0].tolist(); dists = dists[0].tolist()
        out = []
        for j, d in zip(nbrs, dists):
            if j == i or j < 0: 
                continue
            out.append((j, 1.0 - float(d)))
        return out
    return query, index

def topk_brute(emb: np.ndarray, topk: int, block: int = 5000):
    N, D = emb.shape
    E_T = emb.T.copy()
    def query(i):
        best = []
        for start in range(0, N, block):
            end = min(start + block, N)
            sims = emb[i] @ E_T[:, start:end]
            for off, s in enumerate(sims):
                j = start + off
                if j == i: 
                    continue
                if len(best) < topk:
                    best.append((float(s), j))
                    if len(best) == topk:
                        best.sort()
                else:
                    if s > best[0][0]:
                        best[0] = (float(s), j)
                        best.sort()
        best.sort(reverse=True)
        return [(j, s) for (s, j) in best]
    return query, None

def build_query_fn(emb: np.ndarray, topk: int, ann_choice: str):
    ann_choice = (ann_choice or "auto").lower()
    if ann_choice in ("auto","annoy") and ANNOY_OK:
        print("ANN backend: Annoy")
        return topk_annoy(emb, topk)
    if ann_choice in ("auto","pynndescent","pynn") and PYNN_OK:
        print("ANN backend: PyNNDescent")
        return topk_pynnd(emb, topk)
    print("ANN backend: brute (chunked)")
    return topk_brute(emb, topk)

# --------------------------- main ---------------------------

def main():
    print("\n=== GND Polysemy Similarity (per-term counts) ===")

    in_path = ask("Path to GND subjects JSON", "GND-subjects.json")
    if not os.path.isfile(in_path):
        print(f"ERROR: file not found: {in_path}")
        sys.exit(1)
    out_dir = ask("Output directory", "./embeddings")
    ensure_dir(out_dir)

    print("\nChoose view:")
    print("  [1] Name only")
    print("  [2] Name + (Definition + Alternate Names when present)")
    view_choice = ask("View 1/2", "1")
    view = "name" if view_choice.strip() == "1" else "context"

    print("\nChoose encoder backend:")
    print("  [1] HuggingFace (HF)")
    print("  [2] Sentence-Transformers (SBERT)")
    be_choice = ask("Backend 1/2", "1")
    backend = "hf" if be_choice.strip() == "1" else "sbert"

    default_model = "bert-base-multilingual-uncased" if backend == "hf" else "paraphrase-multilingual-MiniLM-L12-v2"
    model_name = ask("Model name", default_model)

    device_kind, device_obj = pick_device_interactive()

    thresh = parse_threshold(ask("Similarity threshold (e.g., 0.9 or 90%)", "0.90"))
    topk = int(ask("Nearest neighbors per item before thresholding (topK)", "50"))

    if backend == "sbert" and not SBERT_OK:
        print("Sentence-Transformers missing. Install with: pip install sentence-transformers")
        sys.exit(1)

    # Load subjects
    print("\nLoading subjects …")
    subjects = load_subjects(in_path)

    # Skip acronym-like names
    keep = [not is_acronymish(s.get("Name","")) for s in subjects]
    kept_idx = [i for i, k in enumerate(keep) if k]
    if not kept_idx:
        print("All subjects were filtered as acronyms; nothing to process.")
        sys.exit(0)

    # Build texts
    names = [subjects[i]["Name"] for i in kept_idx]
    codes = [subjects[i]["Code"] for i in kept_idx]
    if view == "name":
        texts = names
    else:
        texts = [build_context_text(subjects[i]) for i in kept_idx]

    # Encode (cache)
    slug = slugify(model_name)
    npy_path = os.path.join(out_dir, f"{slug}__{view}_embeddings.npy")
    if os.path.isfile(npy_path) and os.path.getsize(npy_path) > 0:
        print(f"Reusing cached embeddings: {npy_path}")
        emb = np.load(npy_path, mmap_mode="r")
        emb = np.asarray(emb, dtype="float32")
        emb = l2_normalize(emb, axis=1)
    else:
        print(f"Encoding with {backend} → {model_name}")
        t0 = time.perf_counter()
        if backend == "hf":
            maxlen = 32 if view == "name" else 128
            emb = encode_hf(model_name, texts, batch_size=128, max_length=maxlen, device_kind=device_kind, device_obj=device_obj)
        else:
            emb = encode_sbert(model_name, texts, batch_size=256, device_kind=device_kind)
        np.save(npy_path, emb.astype("float32"))
        dt = time.perf_counter() - t0
        print(f"Encoded {len(texts):,} items in {dt:.1f}s → {npy_path}")

    # NN search
    query_fn, ann_obj = build_query_fn(emb, topk=topk, ann_choice="auto")

    # Build undirected edge set with threshold
    N = emb.shape[0]
    edges = set()  # store pairs as (min,max)
    print(f"\nFinding neighbors @ threshold ≥ {thresh:.2f} …")
    it = range(N)
    if TQDM_OK:
        it = tqdm(it, total=N, dynamic_ncols=True, desc="NN query")
    for i in it:
        for j, sim in query_fn(i):
            if sim >= thresh:
                a, b = (i, j) if i < j else (j, i)
                if a != b:
                    edges.add((a, b))

    print(f"Edges found: {len(edges):,}")

    # Per-node adjacency & counts
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    counts = [len(v) for v in adj]

    # Write per-term CSV
    th_tag = int(round(thresh * 100))
    counts_path = os.path.join(out_dir, f"similar_counts__{slug}__{view}__th{th_tag}.csv")
    rows = []
    for i in range(N):
        neigh = adj[i]
        rows.append({
            "Name": names[i],
            "GND_ID": codes[i],
            "Similar_Count": len(neigh),
            "Similar_Names": " | ".join(names[j] for j in neigh),
            "Similar_GND_IDs": ";".join(codes[j] for j in neigh),
        })
    pd.DataFrame(rows).to_csv(counts_path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {counts_path}")

    # Summary CSV (metrics + degree histogram)
    degrees = np.array(counts, dtype=np.int32)
    metrics = [
        ("total_subjects_considered", N),
        ("threshold", thresh),
        ("model", model_name),
        ("backend", backend),
        ("view", view),
        ("ann_backend", "annoy" if ANNOY_OK else ("pynndescent" if PYNN_OK else "brute")),
        ("topk", topk),
        ("total_edges", len(edges)),
        ("subjects_with_any_similar", int((degrees > 0).sum())),
        ("mean_degree", float(degrees.mean())),
        ("median_degree", float(np.median(degrees))),
        ("p95_degree", float(np.quantile(degrees, 0.95))),
        ("max_degree", int(degrees.max() if N else 0)),
    ]
    # Flatten histogram as metric rows: degree_k -> count
    unique_deg, counts_deg = np.unique(degrees, return_counts=True)
    for d, c in zip(unique_deg.tolist(), counts_deg.tolist()):
        metrics.append((f"degree_{d}", c))

    summ_path = os.path.join(out_dir, f"similar_summary__{slug}__{view}__th{th_tag}.csv")

    pd.DataFrame(metrics, columns=["metric","value"]).to_csv(summ_path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {summ_path}")

    # Cleanup ANN
    try:
        del ann_obj
    except Exception:
        pass

    print("\nDone.")

if __name__ == "__main__":
    main()
