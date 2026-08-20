"""
Matched-Memory Comparison: MUVERA vs TCI at Equal Storage.
Addresses R2's concern: "How does MUVERA behave with the same memory budget?"

TCI-32 stores: 32 × 128 × 4 bytes = 16 KB/doc
MUVERA default: R=10, B=2 → 20 × 128 × 4 = 10 KB/doc

To match TCI's 16 KB budget, we can increase MUVERA's FDE:
  - R=12, B=2 → 24 × 128 × 4 = 12.3 KB  (still under)
  - R=16, B=2 → 32 × 128 × 4 = 16.4 KB  (matched)
  - R=10, B=3 → 30 × 128 × 4 = 15.4 KB  (close)
  - R=12, B=3 → 36 × 128 × 4 = 18.4 KB  (over)

We compare:
  1. MUVERA (R=10, B=2): 10 KB/doc — default
  2. MUVERA (R=16, B=2): 16 KB/doc — matched to TCI
  3. MUVERA (R=10, B=3): 15 KB/doc — alternative matched
  4. MUVERA (R=20, B=2): 20 KB/doc — generous budget
  5. TCI-32:             16 KB/doc — our method

All use W=200 Chamfer reranks for fair comparison.

Usage:
  python run_matched_memory.py --embeddings-dir data/scifact_colbertv2
  python run_matched_memory.py --embeddings-dir data/nfcorpus_colbertv2
  python run_matched_memory.py --embeddings-dir data/vidore_v3_finance_colpali
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


def encode_fde(doc_tokens, R=10, B=2, seed=42):
    rng = np.random.RandomState(seed)
    dim = doc_tokens.shape[1]
    fde = np.zeros(R * B * dim, dtype=np.float32)
    for r in range(R):
        assignments = rng.randint(0, B, size=len(doc_tokens))
        for b in range(B):
            mask = assignments == b
            if mask.any():
                start = (r * B + b) * dim
                fde[start:start+dim] = doc_tokens[mask].mean(axis=0)
    return fde


def encode_fde_query(q_tokens, R=10, B=2, seed=42):
    rng = np.random.RandomState(seed)
    dim = q_tokens.shape[1]
    fde = np.zeros(R * B * dim, dtype=np.float32)
    for r in range(R):
        assignments = rng.randint(0, B, size=len(q_tokens))
        for b in range(B):
            mask = assignments == b
            if mask.any():
                start = (r * B + b) * dim
                fde[start:start+dim] = q_tokens[mask].sum(axis=0) / R
    return fde


def run_muvera_pipeline(query_tokens, corpus_tokens, ground_truth,
                        R, B, W_prime, W):
    """Run MUVERA pipeline with given FDE config."""
    dim = corpus_tokens[0].shape[1]
    storage_kb = R * B * dim * 4 / 1024

    # Encode FDEs
    doc_fdes = np.array([encode_fde(d, R=R, B=B) for d in corpus_tokens])

    recalls_10, recalls_100, ndcgs, mrrs = [], [], [], []

    for qi in range(len(query_tokens)):
        gt = ground_truth[qi]
        if not gt:
            continue

        q_tok = query_tokens[qi]
        gt_set = set(gt)
        n_rel = len(gt_set)

        q_fde = encode_fde_query(q_tok, R=R, B=B)
        fde_scores = doc_fdes @ q_fde
        top_W_prime = np.argsort(-fde_scores)[:W_prime]

        # Chamfer rerank top-W
        candidates = top_W_prime[:W]
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
        'storage_kb': storage_kb,
        'fde_dim': R * B * dim,
        'n_queries': len(recalls_10),
    }


def run_tci_pipeline(query_tokens, corpus_tokens, ground_truth,
                     tci_centroids, doc_fdes_default,
                     W_prime=1000, W=200):
    """Run TCI pipeline."""
    recalls_10, recalls_100, ndcgs, mrrs = [], [], [], []

    for qi in range(len(query_tokens)):
        gt = ground_truth[qi]
        if not gt:
            continue

        q_tok = query_tokens[qi]
        gt_set = set(gt)
        n_rel = len(gt_set)

        # FDE retrieval (default R=10, B=2)
        q_fde = encode_fde_query(q_tok)
        fde_scores = doc_fdes_default @ q_fde
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
        mrrs.append(1.0 / (next((i+1 for i, d in enumerate(ranking) if d in gt_set), len(ranking)+1)))

    K = tci_centroids[0].shape[0] if len(tci_centroids[0].shape) > 1 else 1
    dim = corpus_tokens[0].shape[1]
    storage_kb = K * dim * 4 / 1024

    return {
        'R@10': float(np.mean(recalls_10)),
        'R@100': float(np.mean(recalls_100)),
        'nDCG@10': float(np.mean(ndcgs)),
        'MRR': float(np.mean(mrrs)),
        'storage_kb': storage_kb,
        'n_queries': len(recalls_10),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--W-prime", type=int, default=1000)
    parser.add_argument("--W", type=int, default=200)
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, name = load_embeddings(args.embeddings_dir)
    dim = corpus_tokens[0].shape[1]

    print("=" * 70)
    print(f"MATCHED-MEMORY COMPARISON: {name}")
    print(f"  Docs: {len(corpus_tokens)}, Queries: {len(query_tokens)}, Dim: {dim}")
    print("=" * 70)

    # MUVERA configurations to test
    fde_configs = [
        (10, 2, "MUVERA(R=10,B=2) [default]"),
        (16, 2, "MUVERA(R=16,B=2) [matched]"),
        (10, 3, "MUVERA(R=10,B=3) [close]"),
        (20, 2, "MUVERA(R=20,B=2) [generous]"),
    ]

    results = {"dataset": name, "n_docs": len(corpus_tokens),
               "n_queries": len(query_tokens), "dim": dim}

    print(f"\n{'Method':<30} {'KB/doc':>8} {'R@10':>8} {'R@100':>8} {'nDCG@10':>8} {'MRR':>8}")
    print("-" * 75)

    # Run MUVERA at different budgets
    for R, B, label in fde_configs:
        print(f"  Running {label}...")
        result = run_muvera_pipeline(
            query_tokens, corpus_tokens, ground_truth,
            R=R, B=B, W_prime=args.W_prime, W=args.W
        )
        print(f"  {label:<28} {result['storage_kb']:>8.1f} "
              f"{result['R@10']:>8.4f} {result['R@100']:>8.4f} "
              f"{result['nDCG@10']:>8.4f} {result['MRR']:>8.4f}")
        results[f"muvera_R{R}_B{B}"] = result

    # Build TCI index
    print(f"\n  Building TCI-{args.K} index...")
    tci_centroids = []
    for tokens in tqdm(corpus_tokens, desc="  TCI", leave=False):
        k = min(args.K, len(tokens))
        if k < 2:
            tci_centroids.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        tci_centroids.append(km.cluster_centers_.astype(np.float32))

    # Default FDE for TCI's first stage
    doc_fdes_default = np.array([encode_fde(d) for d in corpus_tokens])

    # Run TCI pipeline
    print(f"  Running TCI-{args.K}...")
    tci_result = run_tci_pipeline(
        query_tokens, corpus_tokens, ground_truth,
        tci_centroids, doc_fdes_default,
        W_prime=args.W_prime, W=args.W
    )
    # TCI total storage = FDE (10KB) + centroids (16KB) = 26KB
    tci_total_kb = results['muvera_R10_B2']['storage_kb'] + tci_result['storage_kb']
    print(f"  {'TCI-' + str(args.K) + ' (+FDE)':<28} {tci_total_kb:>8.1f} "
          f"{tci_result['R@10']:>8.4f} {tci_result['R@100']:>8.4f} "
          f"{tci_result['nDCG@10']:>8.4f} {tci_result['MRR']:>8.4f}")
    tci_result['total_storage_kb'] = tci_total_kb
    results['tci'] = tci_result

    # Summary
    print("\n" + "=" * 70)
    print("MATCHED-MEMORY SUMMARY")
    print("=" * 70)
    print(f"  MUVERA default (10 KB):    R@100 = {results['muvera_R10_B2']['R@100']:.4f}")
    print(f"  MUVERA matched (16 KB):    R@100 = {results['muvera_R16_B2']['R@100']:.4f}")
    print(f"  MUVERA generous (20 KB):   R@100 = {results['muvera_R20_B2']['R@100']:.4f}")
    print(f"  TCI-{args.K} (10+16 = 26 KB): R@100 = {tci_result['R@100']:.4f}")
    print(f"\n  At matched storage (16 KB), MUVERA R@100 = {results['muvera_R16_B2']['R@100']:.4f}")
    print(f"  TCI uses 26 KB total but achieves R@100 = {tci_result['R@100']:.4f}")
    gain = tci_result['R@100'] - results['muvera_R16_B2']['R@100']
    print(f"  TCI advantage at 1.6× storage: {gain:+.4f} R@100")

    out_file = f"matched_memory_{name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
