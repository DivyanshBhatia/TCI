"""
Encode BEIR datasets with XTR (google/xtr-base-en) for TCI evaluation.

XTR uses T5's ENCODER only to produce per-token embeddings for MaxSim scoring.
We load the full T5Model but call only model.encoder, avoiding the decoder.

Usage (Colab with GPU):
  1. Set DATASET below
  2. Run all cells
  3. Download the zip
  4. Locally: python run_significance.py --embeddings-dir scifact_xtr --tci-k 32

Requirements:
  pip install transformers torch beir numpy tqdm
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install transformers torch beir numpy tqdm

# ============================================================
# CELL 2: Configuration
# ============================================================
import os, json, torch, numpy as np
from tqdm import tqdm

# ---- CHOOSE DATASET ----
# "scifact", "nfcorpus", "fiqa", "scidocs", "arguana", "trec-covid"
DATASET = "scifact"

MODEL_ID = "google/xtr-base-en"
MAX_DOC_LENGTH = 512
MAX_QUERY_LENGTH = 64
BATCH_SIZE_DOC = 32
BATCH_SIZE_QUERY = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model: {MODEL_ID}")
print(f"Dataset: {DATASET}")
print(f"Device: {device}")

# ============================================================
# CELL 3: Load XTR model (encoder only)
# ============================================================
from transformers import AutoTokenizer, T5EncoderModel

print(f"\nLoading {MODEL_ID} (encoder only)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# T5EncoderModel loads ONLY the encoder weights, ignoring decoder
# This avoids the "decoder_input_ids" error
model = T5EncoderModel.from_pretrained(MODEL_ID)
model = model.to(device)
model.eval()

embed_dim = model.config.d_model
print(f"  Embedding dim: {embed_dim}")
print(f"  Loaded successfully")

# ============================================================
# CELL 4: Encoding functions
# ============================================================

def encode_texts(texts, max_length, batch_size, desc="Encoding"):
    """Encode texts to per-token embeddings using XTR encoder."""
    all_embeddings = []
    all_lengths = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        with torch.no_grad():
            # T5EncoderModel.forward() only runs the encoder
            outputs = model(**inputs)

        # outputs.last_hidden_state: (batch, seq_len, dim)
        token_embeds = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # L2 normalize (standard for MaxSim scoring)
        token_embeds = torch.nn.functional.normalize(token_embeds, p=2, dim=-1)

        for i in range(len(batch_texts)):
            mask = attention_mask[i].bool()
            doc_emb = token_embeds[i][mask].cpu().numpy().astype(np.float32)
            all_embeddings.append(doc_emb)
            all_lengths.append(len(doc_emb))

    return all_embeddings, all_lengths


def encode_docs(texts, batch_size=BATCH_SIZE_DOC):
    return encode_texts(texts, MAX_DOC_LENGTH, batch_size, desc="Encoding docs")


def encode_queries(texts, batch_size=BATCH_SIZE_QUERY):
    return encode_texts(texts, MAX_QUERY_LENGTH, batch_size, desc="Encoding queries")


# ============================================================
# CELL 5: Load BEIR dataset
# ============================================================
print(f"\nLoading BEIR dataset: {DATASET}")

from beir import util
from beir.datasets.data_loader import GenericDataLoader

beir_map = {
    "scifact": "scifact",
    "nfcorpus": "nfcorpus",
    "fiqa": "fiqa",
    "scidocs": "scidocs",
    "arguana": "arguana",
    "trec-covid": "trec-covid",
    "quora": "quora",
    "fever": "fever",
    "touche": "webis-touche2020",
    "msmarco": "msmarco",
}

dataset_name = beir_map.get(DATASET, DATASET)
data_path = f"./beir_data/{dataset_name}"

if not os.path.exists(data_path):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    print(f"  Downloading from {url}")
    data_path = util.download_and_unzip(url, "./beir_data")

split = "dev" if DATASET == "msmarco" else "test"
corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

did_list = sorted(corpus.keys())
did_to_idx = {d: i for i, d in enumerate(did_list)}
qid_list = sorted(queries.keys())

doc_texts = [
    (corpus[d].get('title', '') + ' ' + corpus[d].get('text', '')).strip()
    for d in did_list
]
query_texts_list = [queries[qid] for qid in qid_list]

ground_truth = []
for qid in qid_list:
    rel = [
        did_to_idx[d]
        for d in qrels.get(qid, {})
        if qrels[qid][d] > 0 and d in did_to_idx
    ]
    ground_truth.append(rel)

n_with_gt = sum(1 for gt in ground_truth if len(gt) > 0)
print(f"  {len(doc_texts)} docs, {len(query_texts_list)} queries, {n_with_gt} with ground truth")

# ============================================================
# CELL 6: Encode
# ============================================================
print(f"\nEncoding with {MODEL_ID}...")

doc_embs, doc_lengths = encode_docs(doc_texts)
query_embs, query_lengths = encode_queries(query_texts_list)

print(f"  Docs: {len(doc_embs)}, avg {np.mean(doc_lengths):.0f} tokens")
print(f"  Queries: {len(query_embs)}, avg {np.mean(query_lengths):.0f} tokens")
print(f"  Dim: {doc_embs[0].shape[1]}")

# ============================================================
# CELL 7: Save
# ============================================================
out_dir = f"./{DATASET}_xtr"
os.makedirs(out_dir, exist_ok=True)

corpus_flat = np.vstack(doc_embs).astype(np.float32)
query_flat = np.vstack(query_embs).astype(np.float32)

np.save(f"{out_dir}/corpus_flat.npy", corpus_flat)
np.save(f"{out_dir}/corpus_lengths.npy", np.array(doc_lengths, dtype=np.int32))
np.save(f"{out_dir}/query_flat.npy", query_flat)
np.save(f"{out_dir}/query_lengths.npy", np.array(query_lengths, dtype=np.int32))

with open(f"{out_dir}/qrels.json", 'w') as f:
    json.dump({'ground_truth': ground_truth}, f)

print(f"\nSaved to {out_dir}/")
print(f"  Corpus flat: {corpus_flat.shape}")
print(f"  Query flat:  {query_flat.shape}")
print(f"  Corpus lengths: min={min(doc_lengths)}, max={max(doc_lengths)}, avg={np.mean(doc_lengths):.0f}")
print(f"  Query lengths:  min={min(query_lengths)}, max={max(query_lengths)}, avg={np.mean(query_lengths):.0f}")

# Zip for download
import shutil
shutil.make_archive(out_dir, 'zip', '.', f'{DATASET}_xtr')
print(f"  Zipped: {out_dir}.zip")

print(f"\n{'='*60}")
print(f"NEXT STEPS (run locally):")
print(f"{'='*60}")
print(f"  python run_significance.py --embeddings-dir {DATASET}_xtr --tci-k 32")
print(f"  python run_baselines.py --embeddings-dir {DATASET}_xtr")
print(f"")
print(f"Note: XTR uses d={embed_dim}, not 128 like ColBERT.")
print(f"TCI is dimension-agnostic so this works fine.")
