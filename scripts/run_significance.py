"""
Statistical significance tests for TCI vs MUVERA.
Paired tests on per-query Recall@10 and Recall@100.

Usage:
  python run_significance.py --embeddings-dir data/nfcorpus_colbertv2

Computes:
  - Paired t-test (parametric)
  - Wilcoxon signed-rank test (non-parametric)
  - Bootstrap confidence intervals for mean improvement
"""

import argparse
import json
import os
import time
import numpy as np
from scipy import stats
from pathlib import Path


def load_embeddings(embeddings_dir):
    """Load embeddings and ground truth."""
    corpus_flat = np.load(os.path.join(embeddings_dir, "corpus_flat.npy"))
    corpus_lengths = np.load(os.path.join(embeddings_dir, "corpus_lengths.npy"))
    query_flat = np.load(os.path.join(embeddings_dir, "query_flat.npy"))
    query_lengths = np.load(os.path.join(embeddings_dir, "query_lengths.npy"))

    with open(os.path.join(embeddings_dir, "qrels.json")) as f:
        qrels = json.load(f)

    # Reconstruct per-doc and per-query token arrays
    corpus_tokens = []
    offset = 0
    for l in corpus_lengths:
        corpus_tokens.append(corpus_flat[offset:offset+l])
        offset += l

    query_tokens = []
    offset = 0
    for l in query_lengths:
        query_tokens.append(query_flat[offset:offset+l])
        offset += l

    ground_truth = qrels["ground_truth"]
    dataset_name = os.path.basename(embeddings_dir.rstrip("/"))

    return corpus_tokens, query_tokens, ground_truth, dataset_name, corpus_lengths


def chamfer_score(q_tokens, d_tokens):
    """Compute Chamfer similarity (sum of max similarities)."""
    sim = q_tokens @ d_tokens.T  # (n_q, n_d)
    return float(sim.max(axis=1).sum())


def encode_fde(doc_tokens, R=10, seed=42):
    """Encode document tokens to FDE (MUVERA encoding: doc=mean per bucket)."""
    rng = np.random.RandomState(seed)
    dim = doc_tokens.shape[1]
    total_dim = R * 2 * dim  # R reps × 2 buckets × dim

    fde = np.zeros(total_dim, dtype=np.float32)
    n_tokens = len(doc_tokens)

    for r in range(R):
        assignments = rng.randint(0, 2, size=n_tokens)
        for b in range(2):
            mask = assignments == b
            if mask.any():
                bucket_mean = doc_tokens[mask].mean(axis=0)
                start = (r * 2 + b) * dim
                fde[start:start+dim] = bucket_mean

    return fde


def encode_query_fde(query_tokens, R=10, seed=42):
    """Encode query tokens to FDE (MUVERA encoding: query=sum per bucket)."""
    rng = np.random.RandomState(seed)
    dim = query_tokens.shape[1]
    total_dim = R * 2 * dim

    fde = np.zeros(total_dim, dtype=np.float32)
    n_tokens = len(query_tokens)

    for r in range(R):
        assignments = rng.randint(0, 2, size=n_tokens)
        for b in range(2):
            mask = assignments == b
            if mask.any():
                bucket_sum = query_tokens[mask].sum(axis=0)
                start = (r * 2 + b) * dim
                fde[start:start+dim] = bucket_sum / R

    return fde


def build_tci_index(doc_tokens_list, K=32):
    """Build TCI centroids for each document."""
    from sklearn.cluster import KMeans

    tci_centroids = []
    for tokens in doc_tokens_list:
        n = len(tokens)
        k = min(K, n)
        if k < 2:
            tci_centroids.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        tci_centroids.append(km.cluster_centers_.astype(np.float32))

    return tci_centroids


def tci_score(q_tokens, centroids):
    """Approximate Chamfer via TCI centroids."""
    sim = q_tokens @ centroids.T
    return float(sim.max(axis=1).sum())


def per_query_recall(ranked_indices, ground_truth_set, k):
    """Compute recall@k for a single query."""
    if not ground_truth_set:
        return None  # skip queries with no ground truth
    hits = sum(1 for idx in ranked_indices[:k] if idx in ground_truth_set)
    return hits / len(ground_truth_set)


def run(args):
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)

    corpus_tokens, query_tokens, ground_truth, dataset_name, corpus_lengths = \
        load_embeddings(args.embeddings_dir)

    n_docs = len(corpus_tokens)
    n_queries = len(query_tokens)
    avg_tokens = float(np.mean(corpus_lengths))

    print(f"  {n_docs} docs (avg {avg_tokens:.0f} tok), {n_queries} queries")

    # Filter queries with ground truth
    valid_queries = [i for i in range(n_queries) if len(ground_truth[i]) > 0]
    print(f"  {len(valid_queries)} queries with ground truth")

    W = args.rerank_budget  # Chamfer rerank budget
    W_prime = args.fde_candidates  # FDE candidates for TCI
    K = args.tci_k

    print(f"\n  Config: FDE top-{W_prime} → TCI-{K} top-{W} → Chamfer rerank")
    print(f"  Baseline: FDE top-{W} → Chamfer rerank")

    # Step 1: Encode FDEs
    print("\nEncoding FDEs...")
    doc_fdes = np.array([encode_fde(d, R=10) for d in corpus_tokens])
    query_fdes = np.array([encode_query_fde(q, R=10) for q in query_tokens])
    print(f"  Doc FDEs: {doc_fdes.shape}, Query FDEs: {query_fdes.shape}")

    # Step 2: Build TCI index
    print(f"\nBuilding TCI-{K} index...")
    tci_centroids = build_tci_index(corpus_tokens, K=K)
    print(f"  Done")

    # Step 3: Per-query evaluation
    print(f"\nEvaluating {len(valid_queries)} queries...")

    muvera_r10 = []
    muvera_r100 = []
    tci_r10 = []
    tci_r100 = []

    from tqdm import tqdm
    for qi in tqdm(valid_queries, desc="Queries"):
        gt_set = set(ground_truth[qi])
        q_fde = query_fdes[qi]
        q_tok = query_tokens[qi]

        # FDE scores for all docs
        fde_scores = doc_fdes @ q_fde

        # === MUVERA baseline: FDE top-W → Chamfer ===
        muvera_candidates = np.argsort(-fde_scores)[:W]
        muvera_chamfer = []
        for di in muvera_candidates:
            score = chamfer_score(q_tok, corpus_tokens[di])
            muvera_chamfer.append((di, score))
        muvera_chamfer.sort(key=lambda x: -x[1])
        muvera_ranked = [x[0] for x in muvera_chamfer]

        r10 = per_query_recall(muvera_ranked, gt_set, 10)
        r100 = per_query_recall(muvera_ranked, gt_set, 100)
        muvera_r10.append(r10)
        muvera_r100.append(r100)

        # === TCI: FDE top-W' → TCI rescore top-W → Chamfer ===
        tci_candidates_wide = np.argsort(-fde_scores)[:W_prime]

        # TCI rescore
        tci_scores = []
        for di in tci_candidates_wide:
            score = tci_score(q_tok, tci_centroids[di])
            tci_scores.append((di, score))
        tci_scores.sort(key=lambda x: -x[1])
        tci_top_W = [x[0] for x in tci_scores[:W]]

        # Chamfer rerank
        tci_chamfer = []
        for di in tci_top_W:
            score = chamfer_score(q_tok, corpus_tokens[di])
            tci_chamfer.append((di, score))
        tci_chamfer.sort(key=lambda x: -x[1])
        tci_ranked = [x[0] for x in tci_chamfer]

        r10 = per_query_recall(tci_ranked, gt_set, 10)
        r100 = per_query_recall(tci_ranked, gt_set, 100)
        tci_r10.append(r10)
        tci_r100.append(r100)

    # Convert to arrays
    muvera_r10 = np.array(muvera_r10)
    muvera_r100 = np.array(muvera_r100)
    tci_r10 = np.array(tci_r10)
    tci_r100 = np.array(tci_r100)

    # Step 4: Statistical tests
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {}

    for metric_name, m_scores, t_scores in [
        ("R@10", muvera_r10, tci_r10),
        ("R@100", muvera_r100, tci_r100),
    ]:
        diff = t_scores - m_scores
        mean_diff = np.mean(diff)
        n_improved = np.sum(diff > 0)
        n_hurt = np.sum(diff < 0)
        n_tied = np.sum(diff == 0)

        print(f"\n--- {metric_name} ---")
        print(f"  MUVERA mean: {np.mean(m_scores):.4f}")
        print(f"  TCI mean:    {np.mean(t_scores):.4f}")
        print(f"  Mean Δ:      {mean_diff:+.4f}")
        print(f"  Improved/Hurt/Tied: {n_improved}/{n_hurt}/{n_tied}")

        # Paired t-test
        if np.std(diff) > 0:
            t_stat, t_pval = stats.ttest_rel(t_scores, m_scores)
            print(f"  Paired t-test: t={t_stat:.4f}, p={t_pval:.6f} {'***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else 'ns'}")
        else:
            t_stat, t_pval = 0.0, 1.0
            print(f"  Paired t-test: no variance in differences")

        # Wilcoxon signed-rank test
        nonzero_diff = diff[diff != 0]
        if len(nonzero_diff) > 0:
            w_stat, w_pval = stats.wilcoxon(nonzero_diff, alternative='greater')
            print(f"  Wilcoxon test: W={w_stat:.1f}, p={w_pval:.6f} {'***' if w_pval < 0.001 else '**' if w_pval < 0.01 else '*' if w_pval < 0.05 else 'ns'}")
        else:
            w_stat, w_pval = 0.0, 1.0
            print(f"  Wilcoxon test: all differences are zero")

        # Bootstrap 95% CI
        n_boot = 10000
        boot_means = []
        rng = np.random.RandomState(42)
        for _ in range(n_boot):
            idx = rng.choice(len(diff), size=len(diff), replace=True)
            boot_means.append(np.mean(diff[idx]))
        boot_means = np.array(boot_means)
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)
        print(f"  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

        results[metric_name] = {
            "muvera_mean": float(np.mean(m_scores)),
            "tci_mean": float(np.mean(t_scores)),
            "mean_diff": float(mean_diff),
            "n_improved": int(n_improved),
            "n_hurt": int(n_hurt),
            "n_tied": int(n_tied),
            "paired_t_stat": float(t_stat),
            "paired_t_pval": float(t_pval),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_pval": float(w_pval),
            "bootstrap_ci_lo": float(ci_lo),
            "bootstrap_ci_hi": float(ci_hi),
        }

    # Save
    out_dir = f"./results/significance_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "significance_results.json"), "w") as f:
        json.dump({
            "dataset": dataset_name,
            "n_docs": n_docs,
            "n_queries": len(valid_queries),
            "config": {
                "rerank_budget": W,
                "fde_candidates": W_prime,
                "tci_k": K,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nSaved to {out_dir}/significance_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--rerank-budget", type=int, default=200)
    parser.add_argument("--fde-candidates", type=int, default=1000)
    parser.add_argument("--tci-k", type=int, default=32)
    args = parser.parse_args()
    run(args)
