"""
Compression Baselines with Full Pipeline Metrics.
Addresses R3-W5: "results do not appear to be fully reported in a table."

Runs full retrieval pipeline (not just inversion rates) for each compression method.
Reports R@10, R@100, nDCG@10 in a format ready for a paper table.

Methods:
  1. MUVERA (FDE only, no intermediate stage)
  2. TokenPool-32 (consecutive window averaging = ColPali pooling)
  3. TopNorm-32 (select K tokens with highest L2 norm)
  4. TopCentral-32 (select K tokens closest to document mean)
  5. PCA-32 (project to K principal components)
  6. TCI-32 (per-document k-means)

Usage:
  python run_compression_pipeline.py --embeddings-dir data/scifact_colbertv2
  python run_compression_pipeline.py --embeddings-dir data/nfcorpus_colbertv2
  python run_compression_pipeline.py --embeddings-dir data/vidore_v3_finance_colpali
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


def compute_ndcg(ranking, ground_truth, k=10):
    gt_set = set(ground_truth)
    dcg = sum(1.0 / np.log2(i + 2) for i, did in enumerate(ranking[:k]) if did in gt_set)
    n_rel = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


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


# Compression methods
def compress_token_pool(tokens, K):
    n = len(tokens)
    if n <= K:
        return tokens
    window = n // K
    pooled = []
    for i in range(K):
        start = i * window
        end = start + window if i < K - 1 else n
        pooled.append(tokens[start:end].mean(axis=0))
    return np.array(pooled, dtype=np.float32)


def compress_top_norm(tokens, K):
    n = len(tokens)
    if n <= K:
        return tokens
    norms = np.linalg.norm(tokens, axis=1)
    top_idx = np.argsort(-norms)[:K]
    return tokens[top_idx]


def compress_top_central(tokens, K):
    n = len(tokens)
    if n <= K:
        return tokens
    centroid = tokens.mean(axis=0)
    dists = np.linalg.norm(tokens - centroid, axis=1)
    top_idx = np.argsort(dists)[:K]
    return tokens[top_idx]


def compress_pca(tokens, K):
    n = len(tokens)
    if n <= K:
        return tokens
    mean = tokens.mean(axis=0)
    centered = tokens - mean
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        representatives = Vt[:K] * S[:K, None]
        representatives[0] += mean
        return representatives.astype(np.float32)
    except np.linalg.LinAlgError:
        return tokens[:K]


def compress_kmeans(tokens, K):
    n = len(tokens)
    k = min(K, n)
    if k < 2:
        return tokens
    km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
    km.fit(tokens)
    return km.cluster_centers_.astype(np.float32)


def run_intermediate_pipeline(query_tokens, corpus_tokens, ground_truth,
                              doc_fdes, compressed_docs,
                              W_prime=1000, W=200):
    """Run FDE → intermediate rescore → Chamfer pipeline."""
    recalls_10, recalls_100, ndcgs, mrrs = [], [], [], []

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
        top_W_prime = np.argsort(-fde_scores)[:W_prime]

        # Intermediate rescore
        int_scores = [(di, float((q_tok @ compressed_docs[di].T).max(axis=1).sum()))
                      for di in top_W_prime]
        int_scores.sort(key=lambda x: -x[1])
        candidates = [di for di, _ in int_scores[:W]]

        # Chamfer rerank
        chamfer_results = [(di, chamfer_score(q_tok, corpus_tokens[di]))
                           for di in candidates]
        chamfer_results.sort(key=lambda x: -x[1])
        ranking = [di for di, _ in chamfer_results]

        recalls_10.append(len(set(ranking[:10]) & gt_set) / n_rel)
        recalls_100.append(len(set(ranking[:100]) & gt_set) / n_rel)
        ndcgs.append(compute_ndcg(ranking, gt))
        mrrs.append(1.0 / (next((i+1 for i, d in enumerate(ranking) if d in gt_set), len(ranking)+1)))

    return {
        'R@10': float(np.mean(recalls_10)),
        'R@100': float(np.mean(recalls_100)),
        'nDCG@10': float(np.mean(ndcgs)),
        'MRR': float(np.mean(mrrs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--K", type=int, default=32)
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, name = load_embeddings(args.embeddings_dir)

    print("=" * 70)
    print(f"COMPRESSION PIPELINE COMPARISON: {name}")
    print(f"  Docs: {len(corpus_tokens)}, Queries: {len(query_tokens)}, K: {args.K}")
    print("=" * 70)

    # Encode FDEs
    print("\nEncoding FDEs...")
    doc_fdes = np.array([encode_fde(d) for d in tqdm(corpus_tokens, desc="FDE")])

    # MUVERA baseline (no intermediate stage)
    print("\nRunning MUVERA baseline...")
    muv_r10, muv_r100, muv_ndcg, muv_mrr = [], [], [], []
    for qi in tqdm(range(len(query_tokens)), desc="MUVERA"):
        gt = ground_truth[qi]
        if not gt:
            continue
        q_tok = query_tokens[qi]
        gt_set = set(gt)
        n_rel = len(gt_set)

        q_fde = encode_fde_query(q_tok)
        fde_scores = doc_fdes @ q_fde
        candidates = np.argsort(-fde_scores)[:200]

        chamfer_results = [(di, chamfer_score(q_tok, corpus_tokens[di])) for di in candidates]
        chamfer_results.sort(key=lambda x: -x[1])
        ranking = [di for di, _ in chamfer_results]

        muv_r10.append(len(set(ranking[:10]) & gt_set) / n_rel)
        muv_r100.append(len(set(ranking[:100]) & gt_set) / n_rel)
        muv_ndcg.append(compute_ndcg(ranking, gt))
        muv_mrr.append(1.0 / (next((i+1 for i, d in enumerate(ranking) if d in gt_set), len(ranking)+1)))

    muvera_result = {
        'R@10': float(np.mean(muv_r10)), 'R@100': float(np.mean(muv_r100)),
        'nDCG@10': float(np.mean(muv_ndcg)), 'MRR': float(np.mean(muv_mrr)),
    }

    # Compression methods
    methods = {
        "TokenPool": compress_token_pool,
        "TopNorm": compress_top_norm,
        "TopCentral": compress_top_central,
        "PCA": compress_pca,
        "TCI (k-means)": compress_kmeans,
    }

    results = {"dataset": name, "K": args.K, "muvera": muvera_result}

    print(f"\n{'Method':<20} {'R@10':>8} {'R@100':>8} {'nDCG@10':>8} {'MRR':>8} {'ΔR@100':>8}")
    print("-" * 65)
    print(f"  {'MUVERA@200':<18} {muvera_result['R@10']:>8.4f} "
          f"{muvera_result['R@100']:>8.4f} {muvera_result['nDCG@10']:>8.4f} "
          f"{muvera_result['MRR']:>8.4f} {'---':>8}")

    for method_name, compress_fn in methods.items():
        print(f"\n  Compressing with {method_name}-{args.K}...")
        compressed = [compress_fn(tokens, args.K) for tokens in
                      tqdm(corpus_tokens, desc=f"  {method_name}", leave=False)]

        result = run_intermediate_pipeline(
            query_tokens, corpus_tokens, ground_truth,
            doc_fdes, compressed, W_prime=1000, W=200
        )

        delta = result['R@100'] - muvera_result['R@100']
        print(f"  {method_name + '-' + str(args.K):<18} {result['R@10']:>8.4f} "
              f"{result['R@100']:>8.4f} {result['nDCG@10']:>8.4f} "
              f"{result['MRR']:>8.4f} {delta:>+8.4f}")

        results[method_name] = result

    out_file = f"compression_pipeline_{name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
