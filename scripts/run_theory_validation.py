"""
Theorem 5.2 Validation: TCI Approximation Bound.
Verifies that 0 <= Chamfer(Q,D) - TCI(Q,C_D) <= |Q| * eps_K
across query-document pairs.

Usage:
  python run_theory_validation.py --embeddings-dir data/scifact_colbertv2
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

    return corpus_tokens, query_tokens, qrels["ground_truth"], os.path.basename(embeddings_dir.rstrip("/"))


def chamfer_score(q, d):
    return float((q @ d.T).max(axis=1).sum())


def tci_score(q, c):
    return float((q @ c.T).max(axis=1).sum())


def quantization_radius(tokens, centroids, labels):
    """Max L2 distance from any token to its assigned centroid."""
    dists = np.linalg.norm(tokens - centroids[labels], axis=1)
    return float(dists.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--n-pairs", type=int, default=500)
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, name = load_embeddings(args.embeddings_dir)
    dim = corpus_tokens[0].shape[1]

    print("=" * 70)
    print(f"THEOREM VALIDATION: {name}")
    print(f"  Docs: {len(corpus_tokens)}, Queries: {len(query_tokens)}, Dim: {dim}")
    print("=" * 70)

    results = {"dataset": name, "n_docs": len(corpus_tokens),
               "n_queries": len(query_tokens), "dim": dim}

    # Part 1: Quantization radius vs K
    print("\n--- Quantization Radius vs K ---")
    for K in [4, 8, 16, 32, 64]:
        eps_values = []
        for tokens in corpus_tokens[:500]:
            k = min(K, len(tokens))
            if k < 2:
                continue
            km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
            km.fit(tokens)
            eps = quantization_radius(tokens, km.cluster_centers_, km.labels_)
            eps_values.append(eps)
        eps_arr = np.array(eps_values)
        print(f"  K={K:3d}: eps_max={eps_arr.mean():.4f}, eps_p95={np.percentile(eps_arr, 95):.4f}")
        results[f"eps_K{K}"] = {
            "K": K,
            "eps_max_avg": float(eps_arr.mean()),
            "eps_p95_avg": float(np.percentile(eps_arr, 95)),
        }

    # Part 2: Bound verification
    print("\n--- Bound Verification ---")
    for K in [8, 16, 32]:
        # Collect query-relevant pairs
        pairs = []
        for qi in range(len(query_tokens)):
            for di in ground_truth[qi]:
                if di < len(corpus_tokens):
                    pairs.append((qi, di))
        rng = np.random.RandomState(42)
        if len(pairs) > args.n_pairs:
            idx = rng.choice(len(pairs), size=args.n_pairs, replace=False)
            pairs = [pairs[i] for i in idx]

        violations = 0
        actual_errors, bounds = [], []
        for qi, di in tqdm(pairs, desc=f"K={K}"):
            q = query_tokens[qi]
            d = corpus_tokens[di]
            k = min(K, len(d))
            if k < 2:
                continue
            km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
            km.fit(d)
            c = km.cluster_centers_.astype(np.float32)
            eps = quantization_radius(d, c, km.labels_)

            chamfer = chamfer_score(q, d)
            tci = tci_score(q, c)
            error = chamfer - tci
            bound = len(q) * eps

            actual_errors.append(error)
            bounds.append(bound)
            if error > bound + 1e-6:
                violations += 1

        ae = np.array(actual_errors)
        bd = np.array(bounds)
        tightness = float(ae.mean() / bd.mean()) if bd.mean() > 0 else 0

        print(f"  K={K}: violations={violations}/{len(pairs)}, "
              f"tightness={tightness:.3f}, "
              f"error={ae.mean():.2f}±{ae.std():.2f}, "
              f"bound={bd.mean():.2f}±{bd.std():.2f}")

        results[f"theorem2_K{K}"] = {
            "K": K, "n_pairs": len(pairs), "violations": violations,
            "actual_error_mean": float(ae.mean()),
            "actual_error_max": float(ae.max()),
            "bound_mean": float(bd.mean()),
            "bound_max": float(bd.max()),
            "tightness": tightness,
        }

    out_file = f"theory_validation_{name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
