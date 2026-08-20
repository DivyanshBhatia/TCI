"""
K Ablation on Text Datasets.
Addresses R3-W1: "ablation in Table 8 focuses on two visual datasets."

Runs the full pipeline with K ∈ {4, 8, 16, 32, 64} on text datasets
to show when K=32 is sufficient and when smaller K is acceptable.

Usage:
  python run_k_ablation_text.py --embeddings-dir data/scifact_colbertv2
  python run_k_ablation_text.py --embeddings-dir data/nfcorpus_colbertv2
  python run_k_ablation_text.py --embeddings-dir data/fiqa_colbertv2
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


def build_tci_index(corpus_tokens, K):
    centroids = []
    for tokens in tqdm(corpus_tokens, desc=f"  TCI-{K}", leave=False):
        k = min(K, len(tokens))
        if k < 2:
            centroids.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        centroids.append(km.cluster_centers_.astype(np.float32))
    return centroids


def run_pipeline(query_tokens, corpus_tokens, ground_truth,
                 doc_fdes, tci_centroids, W_prime=1000, W=200):
    """Run TCI pipeline, return per-query metrics."""
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--W-prime", type=int, default=1000)
    parser.add_argument("--W", type=int, default=200)
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, name = load_embeddings(args.embeddings_dir)
    avg_tokens = np.mean([len(d) for d in corpus_tokens])

    print("=" * 70)
    print(f"K ABLATION (TEXT): {name}")
    print(f"  Docs: {len(corpus_tokens)}, Avg |D|: {avg_tokens:.0f}")
    print("=" * 70)

    # Encode FDEs (shared across all K values)
    print("\nEncoding FDEs...")
    doc_fdes = np.array([encode_fde(d) for d in tqdm(corpus_tokens, desc="FDE")])

    # Also run MUVERA baseline (no TCI)
    print("\nRunning MUVERA baseline...")
    muvera_r10, muvera_r100, muvera_ndcg = [], [], []
    for qi in tqdm(range(len(query_tokens)), desc="MUVERA"):
        gt = ground_truth[qi]
        if not gt:
            continue
        q_tok = query_tokens[qi]
        gt_set = set(gt)
        n_rel = len(gt_set)

        q_fde = encode_fde_query(q_tok)
        fde_scores = doc_fdes @ q_fde
        candidates = np.argsort(-fde_scores)[:args.W]

        chamfer_results = [(di, chamfer_score(q_tok, corpus_tokens[di]))
                           for di in candidates]
        chamfer_results.sort(key=lambda x: -x[1])
        ranking = [di for di, _ in chamfer_results]

        muvera_r10.append(len(set(ranking[:10]) & gt_set) / n_rel)
        muvera_r100.append(len(set(ranking[:100]) & gt_set) / n_rel)
        muvera_ndcg.append(compute_ndcg(ranking, gt))

    muvera_result = {
        'R@10': float(np.mean(muvera_r10)),
        'R@100': float(np.mean(muvera_r100)),
        'nDCG@10': float(np.mean(muvera_ndcg)),
    }

    results = {"dataset": name, "avg_tokens": float(avg_tokens),
               "muvera": muvera_result}

    # K ablation
    K_values = [4, 8, 16, 32, 64]
    if avg_tokens < 64:
        K_values = [4, 8, 16]  # Skip large K for short docs

    print(f"\n{'Method':<20} {'KB/doc':>8} {'R@10':>8} {'R@100':>8} {'nDCG@10':>8} {'ΔR@100':>8}")
    print("-" * 60)
    print(f"  {'MUVERA@200':<18} {'10.0':>8} "
          f"{muvera_result['R@10']:>8.4f} {muvera_result['R@100']:>8.4f} "
          f"{muvera_result['nDCG@10']:>8.4f} {'---':>8}")

    dim = corpus_tokens[0].shape[1]

    for K in K_values:
        print(f"\n  Building TCI-{K}...")
        tci_centroids = build_tci_index(corpus_tokens, K)

        result = run_pipeline(
            query_tokens, corpus_tokens, ground_truth,
            doc_fdes, tci_centroids,
            W_prime=args.W_prime, W=args.W
        )

        storage_kb = K * dim * 4 / 1024
        delta_r100 = result['R@100'] - muvera_result['R@100']
        print(f"  {'TCI-' + str(K) + '@200':<18} {storage_kb:>8.1f} "
              f"{result['R@10']:>8.4f} {result['R@100']:>8.4f} "
              f"{result['nDCG@10']:>8.4f} {delta_r100:>+8.4f}")

        results[f'tci_K{K}'] = result
        results[f'tci_K{K}']['storage_kb'] = storage_kb
        results[f'tci_K{K}']['delta_r100'] = delta_r100

    out_file = f"k_ablation_{name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
