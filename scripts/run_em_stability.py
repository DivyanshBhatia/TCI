"""
E/M Ratio Subsampling Stability Analysis.
Tests whether ρ predictions are reliable with fewer queries (25, 50, 75, 100, 200).

Usage:
  python run_em_stability.py --embeddings-dir data/scifact_colbertv2
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


def chamfer_score(q, d):
    return float((q @ d.T).max(axis=1).sum())


def compute_rho_for_queries(query_indices, query_tokens, corpus_tokens, doc_fdes, ground_truth, K=32):
    """Compute ρ from a subset of queries."""
    fde_errors, margins = [], []
    for qi in query_indices:
        q_tok = query_tokens[qi]
        gt_set = set(ground_truth[qi])
        if not gt_set:
            continue
        q_fde = encode_fde(q_tok)
        fde_scores = doc_fdes @ q_fde
        top_indices = np.argsort(-fde_scores)[:500]
        rel_in_top = [di for di in top_indices if di in gt_set]
        nonrel_in_top = [di for di in top_indices if di not in gt_set]
        if not rel_in_top or not nonrel_in_top:
            continue
        for di_rel in rel_in_top[:3]:
            chamfer_rel = chamfer_score(q_tok, corpus_tokens[di_rel])
            fde_rel = float(fde_scores[di_rel])
            if abs(chamfer_rel) > 1e-8:
                fde_errors.append(abs(fde_rel - chamfer_rel) / abs(chamfer_rel))
            best_neg = max(nonrel_in_top[:10],
                          key=lambda di: chamfer_score(q_tok, corpus_tokens[di]))
            chamfer_neg = chamfer_score(q_tok, corpus_tokens[best_neg])
            if chamfer_rel > chamfer_neg:
                margins.append((chamfer_rel - chamfer_neg) / chamfer_rel)
    if not fde_errors or not margins:
        return None
    return float(np.mean(fde_errors) / np.median(margins))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, name = load_embeddings(args.embeddings_dir)
    print(f"Dataset: {name}, {len(corpus_tokens)} docs, {len(query_tokens)} queries")

    doc_fdes = np.array([encode_fde(d) for d in corpus_tokens])
    valid = [i for i in range(len(query_tokens)) if len(ground_truth[i]) > 0]
    rng = np.random.RandomState(42)

    results = {"dataset": name}
    for n_q in [25, 50, 75, 100, 200]:
        if n_q > len(valid):
            continue
        rhos = []
        correct = 0
        for trial in range(args.n_bootstrap):
            sample = rng.choice(valid, size=n_q, replace=False).tolist()
            rho = compute_rho_for_queries(sample, query_tokens, corpus_tokens,
                                          doc_fdes, ground_truth)
            if rho is not None:
                rhos.append(rho)
        rhos = np.array(rhos)
        print(f"  n={n_q:3d}: ρ = {rhos.mean():.2f} ± {rhos.std():.2f}, "
              f"CI=[{np.percentile(rhos, 2.5):.2f}, {np.percentile(rhos, 97.5):.2f}]")
        results[f"n_{n_q}"] = {
            "mean": float(rhos.mean()), "std": float(rhos.std()),
            "ci_lo": float(np.percentile(rhos, 2.5)),
            "ci_hi": float(np.percentile(rhos, 97.5)),
        }

    out_file = f"em_stability_{name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
