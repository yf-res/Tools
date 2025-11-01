# faiss_persistence.py
import faiss, numpy as np, json, pathlib, pickle

ARTIFACTS_DIR = pathlib.Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

def save_index_and_rows(index, rows, prefix="photos"):
    faiss.write_index(index, str(ARTIFACTS_DIR / f"{prefix}.faiss"))
    # Save rows (list[dict]) alongside the index
    with open(ARTIFACTS_DIR / f"{prefix}_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {prefix}.faiss and {prefix}_rows.jsonl")

def load_index_and_rows(prefix="photos"):
    index = faiss.read_index(str(ARTIFACTS_DIR / f"{prefix}.faiss"))
    rows = []
    p = ARTIFACTS_DIR / f"{prefix}_rows.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return index, rows
