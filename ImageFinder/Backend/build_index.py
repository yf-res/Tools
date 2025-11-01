# build_index.py
import numpy as np, faiss
from faiss_persistence import save_index_and_rows

# Example: vectors is list[np.array] with identical dims
def build_ip_index(vectors: list[np.ndarray]):
    X = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(X)  # good for cosine/IP
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index

# Suppose you already have `vectors` and `rows` from your ingestion pipeline
# vectors: list[np.ndarray]; rows: list[dict]
# index = build_ip_index(vectors)
# save_index_and_rows(index, rows, prefix="photos")
