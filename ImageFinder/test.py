from fastapi import FastAPI, Query
from faiss_persistence import load_index_and_rows
from your_embedding_module import embed  # same embed() used to build the index
import faiss, numpy as np

app = FastAPI()
INDEX, ROWS = load_index_and_rows(prefix="photos")

@app.get("/search")
def search_photos(q: str = Query(..., min_length=2), k: int = 24):
    qv = embed(q).astype("float32").reshape(1, -1)
    faiss.normalize_L2(qv)
    scores, idxs = INDEX.search(qv, k)
    results = []
    for i, s in zip(idxs[0], scores[0]):
        hit = ROWS[i].copy()
        hit["score"] = float(s)
        results.append(hit)
    return results
