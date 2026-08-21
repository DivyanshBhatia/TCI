"""
Encode datasets with different multi-vector models.
Tests TCI generalization across model architectures.

Supported models:
  1. ColBERTv2 (text) — colbert-ir/colbertv2.0
  2. ColBERTv1 (text) — colbert-ir/colbertv1.9
  3. ColPali (visual) — vidore/colpali-v1.3
  4. ColQwen2 (visual) — vidore/colqwen2-v1.0
  5. XTR (text) — google/xtr-base-en
  6. Jina-ColBERT-v2 (text) — jinaai/jina-colbert-v2

# Compatibility patches for version conflicts
try:
    import peft.import_utils; peft.import_utils.is_torchao_available = lambda: False
except: pass

# Fix ColBERT + newer transformers: all_tied_weights_keys missing
try:
    import torch.nn as nn
    if not hasattr(nn.Module, 'all_tied_weights_keys'):
        nn.Module.all_tied_weights_keys = property(lambda self: getattr(self, '_tied_weights_keys', set()))
except: pass

Usage in Colab:
  Set MODEL_NAME and DATASET, run all cells.
  Then run theory+margin locally.
"""

# ============================================================
# CELL 1: Install dependencies (uncomment as needed)
# ============================================================
# !pip install colpali-engine torch datasets pillow
# !pip install colbert-ai  # for ColBERTv1/v2
# !pip install einops  # for Jina-ColBERT

# ============================================================
# CELL 2: Configuration
# ============================================================
import os, json, numpy as np, torch, argparse, sys
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="colbertv2",
                    choices=["colbertv2", "colbertv1", "colpali", "colqwen2", "colqwen2.5", "xtr", "jina-colbert"],
                    help="Model to encode with")
parser.add_argument("--dataset", type=str, default="scifact",
                    help="Dataset name (e.g., scifact, fiqa, scidocs, vidore_v3_finance, docvqa)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: {dataset}_{model})")
parser.add_argument("--doc-batch-size", type=int, default=64)
parser.add_argument("--query-batch-size", type=int, default=16)
args = parser.parse_args()

MODEL_NAME = args.model
DATASET = args.dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET}")
print(f"Device: {device}")

# ============================================================
# CELL 3: Load model
# ============================================================

def load_model(model_name):
    """Load model and return encode functions."""

    if model_name == "colbertv2":
        from colbert.infra import ColBERTConfig
        from colbert.modeling.checkpoint import Checkpoint
        config = ColBERTConfig(doc_maxlen=180, query_maxlen=32)
        ckpt = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config)

        def encode_docs(texts, batch_size=64):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Docs"):
                batch = texts[i:i+batch_size]
                embs = ckpt.docFromText(batch)
                for emb in embs:
                    e = emb.cpu().numpy().astype(np.float32)
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        def encode_queries(texts, batch_size=64):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Queries"):
                batch = texts[i:i+batch_size]
                embs = ckpt.queryFromText(batch)
                for emb in embs:
                    e = emb.cpu().numpy().astype(np.float32)
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        return encode_docs, encode_queries, None, "text"

    elif model_name == "colbertv1":
        from colbert.infra import ColBERTConfig
        from colbert.modeling.checkpoint import Checkpoint
        config = ColBERTConfig(doc_maxlen=180, query_maxlen=32)
        ckpt = Checkpoint("colbert-ir/colbertv1.9", colbert_config=config)

        def encode_docs(texts, batch_size=64):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Docs"):
                batch = texts[i:i+batch_size]
                embs = ckpt.docFromText(batch)
                for emb in embs:
                    e = emb.cpu().numpy().astype(np.float32)
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        def encode_queries(texts, batch_size=64):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Queries"):
                batch = texts[i:i+batch_size]
                embs = ckpt.queryFromText(batch)
                for emb in embs:
                    e = emb.cpu().numpy().astype(np.float32)
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        return encode_docs, encode_queries, None, "text"

    elif model_name in ("colqwen2", "colqwen2.5"):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        if model_name == "colqwen2":
            model_path = "vidore/colqwen2-v1.0"
        else:
            model_path = "vidore/colqwen2.5-v0.2"

        model = ColQwen2.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        processor = ColQwen2Processor.from_pretrained(model_path)
        model.eval()

        def encode_images(images, batch_size=4):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(images), batch_size), desc="Pages"):
                batch = images[i:i+batch_size]
                inputs = processor.process_images(batch).to(device)
                with torch.no_grad():
                    embeddings = model(**inputs)
                for emb in embeddings:
                    e = emb.cpu().float().numpy()
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        def encode_queries(texts, batch_size=16):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Queries"):
                batch = texts[i:i+batch_size]
                inputs = processor.process_queries(batch).to(device)
                with torch.no_grad():
                    embeddings = model(**inputs)
                for emb in embeddings:
                    e = emb.cpu().float().numpy()
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        return encode_images, encode_queries, None, "visual"

    elif model_name == "colpali":
        from colpali_engine.models import ColPali, ColPaliProcessor

        model = ColPali.from_pretrained(
            "vidore/colpali-v1.3",
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")
        model.eval()

        def encode_images(images, batch_size=4):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(images), batch_size), desc="Pages"):
                batch = images[i:i+batch_size]
                inputs = processor.process_images(batch).to(device)
                with torch.no_grad():
                    embeddings = model(**inputs)
                for emb in embeddings:
                    e = emb.cpu().float().numpy()
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        def encode_queries(texts, batch_size=16):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Queries"):
                batch = texts[i:i+batch_size]
                inputs = processor.process_queries(batch).to(device)
                with torch.no_grad():
                    embeddings = model(**inputs)
                for emb in embeddings:
                    e = emb.cpu().float().numpy()
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        return encode_images, encode_queries, None, "visual"

    elif model_name == "xtr":
        from transformers import AutoTokenizer, T5EncoderModel

        tokenizer = AutoTokenizer.from_pretrained("google/xtr-base-en")
        model = T5EncoderModel.from_pretrained("google/xtr-base-en")
        model = model.to(device).eval()

        def encode_docs(texts, batch_size=32):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Docs"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embs = outputs.last_hidden_state
                    mask = inputs["attention_mask"]

                for j in range(len(batch)):
                    length = int(mask[j].sum())
                    e = embs[j, :length].cpu().numpy().astype(np.float32)
                    # L2 normalize each token embedding
                    norms = np.linalg.norm(e, axis=1, keepdims=True)
                    e = e / np.maximum(norms, 1e-8)
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        def encode_queries(texts, batch_size=32):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Queries"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=32, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embs = outputs.last_hidden_state
                    mask = inputs["attention_mask"]

                for j in range(len(batch)):
                    length = int(mask[j].sum())
                    e = embs[j, :length].cpu().numpy().astype(np.float32)
                    # L2 normalize each token embedding
                    norms = np.linalg.norm(e, axis=1, keepdims=True)
                    e = e / np.maximum(norms, 1e-8)
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        return encode_docs, encode_queries, None, "text"

    elif model_name == "jina-colbert":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-colbert-v2", trust_remote_code=True)
        model = AutoModel.from_pretrained("jinaai/jina-colbert-v2", trust_remote_code=True)
        model = model.to(device).eval()

        def encode_docs(texts, batch_size=32):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Docs"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=180, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Jina-ColBERT outputs last_hidden_state
                    embs = outputs.last_hidden_state
                    mask = inputs["attention_mask"]

                for j in range(len(batch)):
                    length = int(mask[j].sum())
                    e = embs[j, :length].cpu().numpy().astype(np.float32)
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        def encode_queries(texts, batch_size=32):
            all_embs, all_lengths = [], []
            for i in tqdm(range(0, len(texts), batch_size), desc="Queries"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=32, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embs = outputs.last_hidden_state
                    mask = inputs["attention_mask"]

                for j in range(len(batch)):
                    length = int(mask[j].sum())
                    e = embs[j, :length].cpu().numpy().astype(np.float32)
                    all_embs.append(e)
                    all_lengths.append(len(e))
            return all_embs, all_lengths

        return encode_docs, encode_queries, None, "text"

    else:
        raise ValueError(f"Unknown model: {model_name}")


print(f"Loading {MODEL_NAME}...")
encode_docs_fn, encode_queries_fn, _, modality = load_model(MODEL_NAME)
print(f"Loaded! Modality: {modality}")

# ============================================================
# CELL 4: Load dataset
# ============================================================
from datasets import load_dataset

def load_vidore_v3(name):
    """Load ViDoRe V3 dataset."""
    hf_map = {
        "vidore_v3_finance": "vidore/vidore_v3_finance_en",
        "vidore_v3_industrial": "vidore/vidore_v3_industrial",
        "vidore_v3_pharma": "vidore/vidore_v3_pharmaceuticals",
        "vidore_v3_cs": "vidore/vidore_v3_computer_science",
        "vidore_v3_physics": "vidore/vidore_v3_physics",
    }
    hf_id = hf_map.get(name, name)

    corpus_ds = load_dataset(hf_id, "corpus", split="test")
    queries_ds = load_dataset(hf_id, "queries", split="test")
    qrels_ds = load_dataset(hf_id, "qrels", split="test")

    images = [row["image"] for row in corpus_ds]
    corpus_ids = [str(row["corpus_id"]) for row in corpus_ds]
    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    query_ids = [str(row["query_id"]) for row in queries_ds]
    query_texts = [row["query"] for row in queries_ds]

    qrels = {}
    for row in qrels_ds:
        qid, cid, score = str(row["query_id"]), str(row["corpus_id"]), int(row["score"])
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][cid] = score

    ground_truth = []
    for qid in query_ids:
        if qid in qrels:
            ground_truth.append([cid_to_idx[c] for c, s in qrels[qid].items()
                                 if s > 0 and c in cid_to_idx])
        else:
            ground_truth.append([])

    return images, query_texts, query_ids, ground_truth, corpus_ids


def load_vidore_v1(name):
    """Load ViDoRe V1 dataset (QA format)."""
    hf_map = {
        "docvqa": "vidore/docvqa_test_subsampled",
        "arxivqa": "vidore/arxivqa_test_subsampled",
        "tabfquad": "vidore/tabfquad_test_subsampled",
        "shiftproject": "vidore/shiftproject_test",
        "infovqa": "vidore/infographicvqa_test_subsampled",
    }
    hf_id = hf_map.get(name, name)
    dataset = load_dataset(hf_id, split="test")

    images, queries, ground_truth = [], [], []
    qrels_map = {}

    for row_idx, row in enumerate(dataset):
        img = row.get("image")
        query = row.get("query")
        if img is None:
            continue
        page_idx = len(images)
        images.append(img)
        if query and isinstance(query, str) and len(query.strip()) > 0:
            q_idx = len(queries)
            queries.append(query.strip())
            qrels_map[q_idx] = {page_idx: 1}

    ground_truth = []
    for qi in range(len(queries)):
        ground_truth.append(list(qrels_map.get(qi, {}).keys()))

    return images, queries, [str(i) for i in range(len(queries))], ground_truth, [str(i) for i in range(len(images))]


def load_beir(name):
    """Load BEIR text dataset."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    split = "dev" if name == "msmarco" else "test"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    data_path = util.download_and_unzip(url, "./data")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

    did_list = sorted(corpus.keys())
    did_to_idx = {d: i for i, d in enumerate(did_list)}
    qid_list = sorted(queries.keys())

    doc_texts = [(corpus[d].get('title', '') + ' ' + corpus[d].get('text', '')).strip()
                 for d in did_list]
    query_texts = [queries[qid] for qid in qid_list]

    ground_truth = []
    for qid in qid_list:
        rel = [did_to_idx[d] for d in qrels.get(qid, {}) if qrels[qid][d] > 0 and d in did_to_idx]
        ground_truth.append(rel)

    return doc_texts, query_texts, qid_list, ground_truth, did_list


# Load based on dataset type
if DATASET.startswith("vidore_v3"):
    doc_data, query_texts, query_ids, ground_truth, corpus_ids = load_vidore_v3(DATASET)
    is_visual = True
elif DATASET in ("docvqa", "arxivqa", "tabfquad", "shiftproject", "infovqa"):
    doc_data, query_texts, query_ids, ground_truth, corpus_ids = load_vidore_v1(DATASET)
    is_visual = True
else:
    doc_data, query_texts, query_ids, ground_truth, corpus_ids = load_beir(DATASET)
    is_visual = False

n_with_gt = sum(1 for gt in ground_truth if len(gt) > 0)
print(f"\nLoaded: {len(doc_data)} docs, {len(query_texts)} queries, {n_with_gt} with ground truth")

# ============================================================
# CELL 5: Encode
# ============================================================
print(f"\nEncoding with {MODEL_NAME}...")

if is_visual:
    doc_embs, doc_lengths = encode_docs_fn(doc_data, batch_size=args.doc_batch_size if args.doc_batch_size != 64 else 4)
else:
    doc_embs, doc_lengths = encode_docs_fn(doc_data, batch_size=args.doc_batch_size)

query_embs, query_lengths = encode_queries_fn(query_texts, batch_size=args.query_batch_size)

print(f"  Docs: {len(doc_embs)}, avg {np.mean(doc_lengths):.0f} vectors")
print(f"  Queries: {len(query_embs)}, avg {np.mean(query_lengths):.0f} vectors")
print(f"  Dim: {doc_embs[0].shape[1]}")

# ============================================================
# CELL 6: Save
# ============================================================
out_dir = args.output_dir if args.output_dir else f"{DATASET}_{MODEL_NAME}"
os.makedirs(out_dir, exist_ok=True)

corpus_flat = np.vstack(doc_embs).astype(np.float32)
query_flat = np.vstack(query_embs).astype(np.float32)

np.save(f"{out_dir}/corpus_flat.npy", corpus_flat)
np.save(f"{out_dir}/corpus_lengths.npy", np.array(doc_lengths, dtype=np.int32))
np.save(f"{out_dir}/query_flat.npy", query_flat)
np.save(f"{out_dir}/query_lengths.npy", np.array(query_lengths, dtype=np.int32))

with open(f"{out_dir}/qrels.json", 'w') as f:
    json.dump({'ground_truth': ground_truth}, f)

print(f"\nSaved to {out_dir}")
print(f"  Corpus: {corpus_flat.shape}")
print(f"  Queries: {query_flat.shape}")

# Optionally zip for download (uncomment if needed)
# import shutil
# shutil.make_archive(out_dir, 'zip', '.', out_dir)
# print(f"  Zipped: {out_dir}.zip")

print(f"\nRun locally:")
print(f"  python run_theory_validation.py --embeddings-dir {DATASET}_{MODEL_NAME}")
print(f"  python run_margin_analysis.py --embeddings-dir {DATASET}_{MODEL_NAME}")
print(f"  python run_significance.py --embeddings-dir {DATASET}_{MODEL_NAME}")
