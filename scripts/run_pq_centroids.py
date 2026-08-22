"""
Product Quantization on TCI Centroids.
Addresses reviewer Q3: "Have you tried PQ/OPQ on centroids?"

Tests TCI quality at different compression levels:
  - Float32:  16 KB/doc (K=32, d=128) — baseline
  - PQ-8bit:   4 KB/doc (4× compression)
  - PQ-4bit:   2 KB/doc (8× compression)
  - PQ-2bit:   1 KB/doc (16× compression)
  - Scalar-8bit: 4 KB/doc (simple quantization)

Usage:
  python run_pq_centroids.py --embeddings-dir data/scifact_colbertv2
  python run_pq_centroids.py --embeddings-dir data/vidore_v3_finance_colpali
"""

import argparse
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def load_embeddings(embeddings_dir):
    corpus_flat = np.load(os.path.join(embeddings_dir, "corpus_flat.npy"))
    corpus_lengths = np.load(os.path.join(embeddings_dir, "corpus_lengths.npy"))
    query_flat = np.load(os.path.join(embeddings_dir, "query_flat.npy"))
    query_lengths = np.load(os.path.join(embeddings_dir, "query_lengths.npy"))
    with open(os.path.join(embeddings_dir, "qrels.json")) as f:
        qrels = json.load(f)

    corpus_tokens, offset = [], 0
    for l in corpus_lengths:
        corpus_tokens.append(corpus_flat[offset:offset+l])
        offset += l

    query_tokens, offset = [], 0
    for l in query_lengths:
        query_tokens.append(query_flat[offset:offset+l])
        offset += l

    return corpus_tokens, query_tokens, qrels["ground_truth"], \
           os.path.basename(embeddings_dir.rstrip("/"))


def chamfer_score(q, d):
    return float((q @ d.T).max(axis=1).sum())


def encode_fde(doc_tokens, R=10, seed=42):
    rng = np.random.RandomState(seed)
    dim = doc_tokens.shape[1]
    fde = np.zeros(R * 2 * dim, dtype=np.float32)
    for r in range(R):
        assignments = rng.randint(0, 2, size=len(doc_tokens))
        for b in range(2):
            mask = assignments == b
            if mask.any():
                start = (r * 2 + b) * dim
                fde[start:start+dim] = doc_tokens[mask].mean(axis=0)
    return fde


def encode_fde_query(q_tokens, R=10, seed=42):
    rng = np.random.RandomState(seed)
    dim = q_tokens.shape[1]
    fde = np.zeros(R * 2 * dim, dtype=np.float32)
    for r in range(R):
        assignments = rng.randint(0, 2, size=len(q_tokens))
        for b in range(2):
            mask = assignments == b
            if mask.any():
                start = (r * 2 + b) * dim
                fde[start:start+dim] = q_tokens[mask].sum(axis=0) / R
    return fde


def compute_ndcg(ranking, ground_truth, k=10):
    gt_set = set(ground_truth)
    dcg = sum(1.0 / np.log2(i + 2) for i, did in enumerate(ranking[:k]) if did in gt_set)
    n_rel = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


# ================================================================
# Quantization methods
# ================================================================

def scalar_quantize(centroids, bits=8):
    """Per-dimension min-max scalar quantization."""
    n_levels = 2 ** bits
    all_c = np.vstack(centroids)
    mins = all_c.min(axis=0)
    maxs = all_c.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-8] = 1.0  # avoid division by zero

    quantized = []
    for c in centroids:
        # Quantize
        normalized = (c - mins) / ranges
        codes = np.clip(np.round(normalized * (n_levels - 1)), 0, n_levels - 1)
        # Dequantize
        dequant = (codes / (n_levels - 1)) * ranges + mins
        quantized.append(dequant.astype(np.float32))

    return quantized


def product_quantize(centroids, n_subvectors=16, bits=8):
    """
    Product quantization: split each vector into n_subvectors subvectors,
    quantize each independently using k-means codebooks.
    
    Storage per centroid: n_subvectors * bits / 8 bytes
    For K=32, d=128:
      PQ-8bit (16 sub, 8 bits): 32 * 16 * 1 = 512 bytes = 0.5 KB/doc
      PQ-4bit (16 sub, 4 bits): 32 * 16 * 0.5 = 256 bytes = 0.25 KB/doc
    Plus codebook: n_subvectors * 2^bits * (d/n_subvectors) * 4 bytes (shared)
    """
    n_codes = 2 ** bits
    all_c = np.vstack(centroids)
    dim = all_c.shape[1]
    sub_dim = dim // n_subvectors

    # Train codebooks on all centroids
    codebooks = []
    for s in range(n_subvectors):
        start = s * sub_dim
        end = start + sub_dim
        sub_vectors = all_c[:, start:end]
        n_clusters = min(n_codes, len(sub_vectors))
        km = KMeans(n_clusters=n_clusters, n_init=1, max_iter=50, random_state=42)
        km.fit(sub_vectors)
        codebooks.append(km.cluster_centers_.astype(np.float32))

    # Quantize and dequantize each centroid set
    quantized = []
    for c in centroids:
        dequant = np.zeros_like(c)
        for s in range(n_subvectors):
            start = s * sub_dim
            end = start + sub_dim
            sub_v = c[:, start:end]
            # Find nearest codebook entry
            dists = np.linalg.norm(sub_v[:, None, :] - codebooks[s][None, :, :], axis=2)
            codes = dists.argmin(axis=1)
            dequant[:, start:end] = codebooks[s][codes]
        quantized.append(dequant.astype(np.float32))

    # Compute storage
    codebook_size = sum(cb.nbytes for cb in codebooks)
    per_doc_bytes = centroids[0].shape[0] * n_subvectors * (bits / 8)

    return quantized, codebook_size, per_doc_bytes


def run_pipeline(query_tokens, corpus_tokens, ground_truth,
                 doc_fdes, tci_centroids, W_prime=1000, W=200):
    """Run TCI pipeline with given centroids."""
    recalls_10, recalls_100, ndcgs = [], [], []

    for qi in range(len(query_tokens)):
        gt = ground_truth[qi]
        if not gt:
            continue

        q_tok = query_tokens[qi]
        gt_set = set(gt)
        n_rel = len(gt_set)

        # FDE retrieval
        q_fde = encode_fde_query(q_tok)
        fde_scores = doc_fdes @ q_fde
        fde_scores = np.nan_to_num(fde_scores, nan=-1e9)
        top_W_prime = np.argsort(-fde_scores)[:W_prime]

        # TCI rescore
        tci_scores = [(di, float((q_tok @ tci_centroids[di].T).max(axis=1).sum()))
                      for di in top_W_prime]
        tci_scores.sort(key=lambda x: -x[1])
        tci_candidates = [di for di, _ in tci_scores[:W]]

        # Chamfer rerank
        chamfer_results = [(di, chamfer_score(q_tok, corpus_tokens[di]))
                           for di in tci_candidates]
        chamfer_results.sort(key=lambda x: -x[1])
        ranking = [di for di, _ in chamfer_results]

        recalls_10.append(len(set(ranking[:10]) & gt_set) / n_rel)
        recalls_100.append(len(set(ranking[:100]) & gt_set) / n_rel)
        ndcgs.append(compute_ndcg(ranking, gt))

    return {
        'R@10': float(np.mean(recalls_10)),
        'R@100': float(np.mean(recalls_100)),
        'nDCG@10': float(np.mean(ndcgs)),
        'n_queries': len(recalls_10),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--K", type=int, default=32)
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, name = load_embeddings(args.embeddings_dir)
    dim = corpus_tokens[0].shape[1]

    print("=" * 70)
    print(f"PQ ON TCI CENTROIDS: {name}")
    print(f"  Docs: {len(corpus_tokens)}, Dim: {dim}, K: {args.K}")
    print("=" * 70)

    # Encode FDEs
    print("\nEncoding FDEs...")
    doc_fdes = np.array([encode_fde(d) for d in tqdm(corpus_tokens, desc="FDE")])

    # Build TCI centroids (float32 baseline)
    print(f"\nBuilding TCI-{args.K} centroids...")
    tci_centroids_f32 = []
    for tokens in tqdm(corpus_tokens, desc="TCI"):
        k = min(args.K, len(tokens))
        if k < 2:
            tci_centroids_f32.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        tci_centroids_f32.append(km.cluster_centers_.astype(np.float32))

    # Storage calculations
    float32_kb = args.K * dim * 4 / 1024
    print(f"\n  Float32 storage: {float32_kb:.1f} KB/doc")

    results = {"dataset": name, "dim": dim, "K": args.K}

    # ================================================================
    # Test configurations
    # ================================================================
    configs = [
        ("Float32 (baseline)", tci_centroids_f32, float32_kb),
    ]

    # Scalar quantization
    for bits in [8, 4]:
        print(f"\n  Scalar-{bits}bit quantization...")
        sq = scalar_quantize(tci_centroids_f32, bits=bits)
        storage = args.K * dim * bits / 8 / 1024
        configs.append((f"Scalar-{bits}bit", sq, storage))

    # Product quantization
    for n_sub, bits in [(16, 8), (16, 4), (32, 8), (32, 4)]:
        print(f"\n  PQ (sub={n_sub}, bits={bits})...")
        pq, cb_size, per_doc = product_quantize(tci_centroids_f32,
                                                 n_subvectors=n_sub, bits=bits)
        storage = per_doc / 1024  # per doc in KB
        label = f"PQ-{bits}bit (sub={n_sub})"
        configs.append((label, pq, storage))

    # ================================================================
    # Run pipelines
    # ================================================================
    print(f"\n{'Method':<25} {'KB/doc':>8} {'R@10':>8} {'R@100':>8} {'nDCG@10':>8}")
    print("-" * 60)

    # MUVERA baseline (no TCI)
    print("  Running MUVERA baseline...")
    muv_r10, muv_r100, muv_ndcg = [], [], []
    for qi in range(len(query_tokens)):
        gt = ground_truth[qi]
        if not gt:
            continue
        q_tok = query_tokens[qi]
        gt_set = set(gt)
        n_rel = len(gt_set)
        q_fde = encode_fde_query(q_tok)
        fde_scores = doc_fdes @ q_fde
        fde_scores = np.nan_to_num(fde_scores, nan=-1e9)
        candidates = np.argsort(-fde_scores)[:200]
        chamfer_results = [(di, chamfer_score(q_tok, corpus_tokens[di])) for di in candidates]
        chamfer_results.sort(key=lambda x: -x[1])
        ranking = [di for di, _ in chamfer_results]
        muv_r10.append(len(set(ranking[:10]) & gt_set) / n_rel)
        muv_r100.append(len(set(ranking[:100]) & gt_set) / n_rel)
        muv_ndcg.append(compute_ndcg(ranking, gt))

    muv_result = {'R@10': float(np.mean(muv_r10)), 'R@100': float(np.mean(muv_r100)),
                  'nDCG@10': float(np.mean(muv_ndcg))}
    print(f"  {'MUVERA (no TCI)':<23} {'10.0':>8} {muv_result['R@10']:>8.4f} "
          f"{muv_result['R@100']:>8.4f} {muv_result['nDCG@10']:>8.4f}")
    results['muvera'] = muv_result

    for label, centroids, storage in configs:
        print(f"  Running {label}...")
        result = run_pipeline(query_tokens, corpus_tokens, ground_truth,
                              doc_fdes, centroids)
        print(f"  {label:<23} {storage:>8.1f} {result['R@10']:>8.4f} "
              f"{result['R@100']:>8.4f} {result['nDCG@10']:>8.4f}")
        results[label] = {**result, 'storage_kb': storage}

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPRESSION SUMMARY")
    print(f"{'=' * 60}")
    base_r100 = results['Float32 (baseline)']['R@100']
    for label, _, storage in configs:
        r100 = results[label]['R@100']
        loss = base_r100 - r100
        compression = float32_kb / max(storage, 0.01)
        print(f"  {label:<23} {storage:>6.1f} KB  {compression:>5.1f}×  "
              f"R@100={r100:.4f}  loss={loss:+.4f}")

    out_file = f"pq_centroids_{name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
