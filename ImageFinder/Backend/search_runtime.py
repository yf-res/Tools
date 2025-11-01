# search_runtime.py
import faiss, numpy as np
from faiss_persistence import load_index_and_rows
from your_embedding_module import embed  # your embed() from the main guide

index, rows = load_index_and_rows(prefix="photos")

def semantic_search(query: str, k: int = 24):
    qv = embed(query).astype("float32").reshape(1, -1)
    faiss.normalize_L2(qv)
    scores, idxs = index.search(qv, k)
    out = []
    for i, s in zip(idxs[0], scores[0]):
        r = rows[i].copy()
        r["score"] = float(s)
        out.append(r)
    return out

def upsert_vectors(index, rows, new_vectors, new_rows):
    # For IndexFlat* types, you cannot remove easily; you can append.
    index.add(np.vstack(new_vectors).astype("float32"))
    rows.extend(new_rows)
    # Re-save artifacts
    from faiss_persistence import save_index_and_rows
    save_index_and_rows(index, rows, prefix="photos")


if __name__ == "__main__":
    hits = semantic_search("kids building a sandcastle at sunset", k=10)
    for h in hits:
        print(h["score"], h["caption"], h.get("web_url"))
