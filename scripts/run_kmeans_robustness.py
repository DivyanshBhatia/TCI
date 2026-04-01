"""
K-means Robustness Analysis for TCI.
Addresses two reviewer concerns:
  W4: Cluster quality comparison across datasets (silhouette scores)
  W5: Sensitivity to k-means random initialization seed

Usage:
  python run_kmeans_robustness.py --embeddings-dir data/scifact_colbertv2
  python run_kmeans_robustness.py --embeddings-dir data/touche_colbertv2
  python run_kmeans_robustness.py --embeddings-dir data/nfcorpus_colbertv2

Run on at least: SciFact (or NFCorpus) + Touche to compare cluster quality.
"""

import argparse
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def load_embeddings(embeddings_dir):
    corpus_flat = np.load(os.path.join(embeddings_dir, "corpus_flat.npy"))
    corpus_lengths = np.load(os.path.join(embeddings_dir, "corpus_lengths.npy"))
    query_flat = np.load(os.path.join(embeddings_dir, "query_flat.npy"))
    query_lengths = np.load(os.path.join(embeddings_dir, "query_lengths.npy"))

    with open(os.path.join(embeddings_dir, "qrels.json")) as f:
        qrels = json.load(f)

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
    sim = q_tokens @ d_tokens.T
    return float(sim.max(axis=1).sum())


def tci_score(q_tokens, centroids):
    sim = q_tokens @ centroids.T
    return float(sim.max(axis=1).sum())


def encode_fde(doc_tokens, R=10, seed=42):
    rng = np.random.RandomState(seed)
    dim = doc_tokens.shape[1]
    total_dim = R * 2 * dim
    fde = np.zeros(total_dim, dtype=np.float32)
    n_tokens = len(doc_tokens)
    for r in range(R):
        assignments = rng.randint(0, 2, size=n_tokens)
        for b in range(2):
            mask = assignments == b
            if mask.any():
                start = (r * 2 + b) * dim
                fde[start:start+dim] = doc_tokens[mask].mean(axis=0)
    return fde


def build_tci_index(corpus_tokens, K=32, seed=42):
    centroids = []
    for tokens in corpus_tokens:
        n = len(tokens)
        k = min(K, n)
        if k < 2:
            centroids.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=seed)
        km.fit(tokens)
        centroids.append(km.cluster_centers_.astype(np.float32))
    return centroids


def compute_inversion_rate(query_tokens, corpus_tokens, ground_truth,
                           tci_centroids, doc_fdes, max_queries=200):
    """Compute pairwise inversion rate for TCI."""
    valid = [i for i in range(len(query_tokens)) if len(ground_truth[i]) > 0]
    if len(valid) > max_queries:
        rng = np.random.RandomState(42)
        valid = sorted(rng.choice(valid, size=max_queries, replace=False).tolist())

    total_pairs = 0
    tci_inversions = 0

    for qi in valid:
        gt_set = set(ground_truth[qi])
        q_tok = query_tokens[qi]

        # Get FDE top candidates
        q_fde = encode_fde(q_tok, R=10, seed=42)
        fde_scores = doc_fdes @ q_fde
        top_indices = np.argsort(-fde_scores)[:200]

        # Chamfer scores for top candidates
        rel_in_top = [di for di in top_indices if di in gt_set]
        nonrel_in_top = [di for di in top_indices if di not in gt_set]

        if not rel_in_top or not nonrel_in_top:
            continue

        for di_rel in rel_in_top[:3]:  # limit for speed
            chamfer_rel = chamfer_score(q_tok, corpus_tokens[di_rel])
            tci_rel = tci_score(q_tok, tci_centroids[di_rel])

            for di_neg in nonrel_in_top[:10]:  # limit for speed
                chamfer_neg = chamfer_score(q_tok, corpus_tokens[di_neg])
                if chamfer_rel <= chamfer_neg:
                    continue  # not a valid pair

                total_pairs += 1
                tci_neg = tci_score(q_tok, tci_centroids[di_neg])
                if tci_rel < tci_neg:
                    tci_inversions += 1

    rate = tci_inversions / max(total_pairs, 1)
    return rate, tci_inversions, total_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of random seeds for variance analysis")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--max-docs-silhouette", type=int, default=500,
                        help="Max docs for silhouette analysis (slow on large corpora)")
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, dataset_name, corpus_lengths = \
        load_embeddings(args.embeddings_dir)

    n_docs = len(corpus_tokens)
    dim = corpus_tokens[0].shape[1]
    avg_tokens = float(np.mean(corpus_lengths))

    print("=" * 70)
    print("K-MEANS ROBUSTNESS ANALYSIS")
    print("=" * 70)
    print(f"  Dataset: {dataset_name}")
    print(f"  Documents: {n_docs}, Dim: {dim}, Avg tokens: {avg_tokens:.0f}")
    print(f"  K: {args.K}, Seeds: {args.n_seeds}")

    results = {
        "dataset": dataset_name,
        "n_docs": n_docs,
        "dim": dim,
        "avg_tokens": avg_tokens,
        "K": args.K,
    }

    # ================================================================
    # PART 1: Cluster Quality (W4)
    # ================================================================
    print(f"\n--- PART 1: Cluster Quality (silhouette scores) ---")

    n_sil = min(n_docs, args.max_docs_silhouette)
    sil_indices = list(range(n_sil))

    silhouette_scores = []
    intra_cluster_sims = []
    n_degenerate = 0  # clusters with < 2 members

    for idx in tqdm(sil_indices, desc="Silhouette"):
        tokens = corpus_tokens[idx]
        n = len(tokens)
        k = min(args.K, n)
        if k < 2 or n < k + 1:
            continue

        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        labels = km.labels_

        # Check for degenerate clusters (single member)
        unique, counts = np.unique(labels, return_counts=True)
        n_singleton = int(np.sum(counts == 1))
        if n_singleton > k * 0.5:
            n_degenerate += 1

        # Silhouette score (cosine)
        try:
            sil = silhouette_score(tokens, labels, metric='cosine')
            silhouette_scores.append(sil)
        except ValueError:
            pass

        # Intra-cluster cosine similarity
        cluster_sims = []
        for c in unique:
            mask = labels == c
            if mask.sum() < 2:
                continue
            cluster_vecs = tokens[mask]
            # Mean pairwise cosine similarity within cluster
            norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
            normed = cluster_vecs / (norms + 1e-8)
            sim_matrix = normed @ normed.T
            n_c = len(cluster_vecs)
            # Mean of upper triangle
            triu_sum = (sim_matrix.sum() - n_c) / max(n_c * (n_c - 1), 1)
            cluster_sims.append(float(triu_sum))

        if cluster_sims:
            intra_cluster_sims.append(float(np.mean(cluster_sims)))

    sil_arr = np.array(silhouette_scores)
    intra_arr = np.array(intra_cluster_sims)

    print(f"\n  Silhouette scores (cosine, n={len(sil_arr)}):")
    print(f"    Mean: {sil_arr.mean():.4f}")
    print(f"    Std:  {sil_arr.std():.4f}")
    print(f"    Median: {np.median(sil_arr):.4f}")
    print(f"    P10/P90: {np.percentile(sil_arr, 10):.4f} / {np.percentile(sil_arr, 90):.4f}")

    print(f"  Intra-cluster cosine similarity (n={len(intra_arr)}):")
    print(f"    Mean: {intra_arr.mean():.4f}")
    print(f"    Std:  {intra_arr.std():.4f}")

    print(f"  Degenerate docs (>50% singleton clusters): {n_degenerate}/{n_sil}")

    results["cluster_quality"] = {
        "n_docs_analyzed": len(sil_arr),
        "silhouette_mean": float(sil_arr.mean()),
        "silhouette_std": float(sil_arr.std()),
        "silhouette_median": float(np.median(sil_arr)),
        "silhouette_p10": float(np.percentile(sil_arr, 10)),
        "silhouette_p90": float(np.percentile(sil_arr, 90)),
        "intra_cluster_sim_mean": float(intra_arr.mean()),
        "intra_cluster_sim_std": float(intra_arr.std()),
        "n_degenerate_docs": n_degenerate,
    }

    # ================================================================
    # PART 2: Seed Variance (W5)
    # ================================================================
    print(f"\n--- PART 2: K-means Seed Variance ({args.n_seeds} seeds) ---")

    # Encode FDEs once (fixed across seeds)
    print("  Encoding FDEs...")
    doc_fdes = np.array([encode_fde(d, R=10, seed=42) for d in corpus_tokens])

    seed_results = []
    for seed in range(args.n_seeds):
        print(f"  Seed {seed}...", end=" ", flush=True)
        tci_centroids = build_tci_index(corpus_tokens, K=args.K, seed=seed)
        inv_rate, inv_count, total_pairs = compute_inversion_rate(
            query_tokens, corpus_tokens, ground_truth,
            tci_centroids, doc_fdes, max_queries=200
        )
        print(f"inversion rate = {inv_rate:.4f} ({inv_count}/{total_pairs})")
        seed_results.append({
            "seed": seed,
            "inversion_rate": inv_rate,
            "inversions": inv_count,
            "total_pairs": total_pairs,
        })

    rates = np.array([r["inversion_rate"] for r in seed_results])
    print(f"\n  Inversion rates across {args.n_seeds} seeds:")
    print(f"    Mean: {rates.mean():.4f}")
    print(f"    Std:  {rates.std():.4f}")
    print(f"    Min:  {rates.min():.4f}")
    print(f"    Max:  {rates.max():.4f}")
    print(f"    Range: {rates.max() - rates.min():.4f}")
    print(f"    CV (std/mean): {rates.std()/max(rates.mean(), 1e-8):.3f}")

    results["seed_variance"] = {
        "n_seeds": args.n_seeds,
        "inversion_rate_mean": float(rates.mean()),
        "inversion_rate_std": float(rates.std()),
        "inversion_rate_min": float(rates.min()),
        "inversion_rate_max": float(rates.max()),
        "inversion_rate_range": float(rates.max() - rates.min()),
        "coefficient_of_variation": float(rates.std() / max(rates.mean(), 1e-8)),
        "per_seed": seed_results,
    }

    # Save
    out_file = f"kmeans_robustness_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
