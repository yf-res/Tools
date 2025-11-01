from fastapi import FastAPI, Query
app = FastAPI()

# Assume you persisted FAISS index + rows to disk and loaded them:
INDEX = ...
ROWS  = ...

@app.get("/search")
def search_photos(q: str = Query(..., min_length=2), k: int = 24):
    hits = search(q, INDEX, ROWS, k=k)
    return [
        {
          "source": h["source"],
          "open_url": h["web_url"],
          "thumb_url": h["thumb_url"],
          "taken_at": h["taken_at"],
          "caption": h["caption"],
          "tags": h["tags"],
          "score": h["score"],
        } for h in hits
    ]
