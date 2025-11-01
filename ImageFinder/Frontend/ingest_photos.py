# ingest_photos.py
import time, numpy as np
from openai import OpenAI
from google_photos_client import list_media_items
from ms_oauth_setup import onedrive_search, onedrive_thumbnails
from faiss_persistence import save_index_and_rows
import faiss

client = OpenAI()  # expects OPENAI_API_KEY env

EMBED_MODEL = "text-embedding-3-large"
VISION_MODEL = "gpt-4.1"  # or your chosen vision-capable model

def embed(text: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(e.data[0].embedding, dtype="float32")

def describe_image(image_url: str) -> dict:
    rsp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role":"system","content":"You write concise, factual photo captions and 5-10 comma-separated tags."},
            {"role":"user","content":[
                {"type":"text","text":"Caption and tags for this photo. Output JSON {caption, tags}."},
                {"type":"image_url","image_url":{"url": image_url}}
            ]}
        ],
        temperature=0.2,
    )
    txt = rsp.choices[0].message.content
    import json
    try:
        return json.loads(txt)
    except Exception:
        return {"caption": txt, "tags": []}

def build_index(vectors):
    X = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(X)
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    return idx

def ingest_google_photos(max_pages=3):
    vectors, rows = [], []
    page_token = None
    for _ in range(max_pages):
        data = list_media_items(page_size=100, page_token=page_token)
        for m in data.get("mediaItems", []):
            base_url = m["baseUrl"]          # append size params for thumbs
            web_url  = m.get("productUrl", base_url)
            meta = {
                "source": "google",
                "item_id": m["id"],
                "web_url": web_url,
                "thumb_url": base_url + "=w512-h512",
                "taken_at": m.get("mediaMetadata", {}).get("creationTime"),
                "album": None,
                "filename": m.get("filename")
            }
            desc = describe_image(base_url)
            caption = desc.get("caption","")
            tags = desc.get("tags", [])
            text = " | ".join(filter(None,[caption, ", ".join(tags), meta["filename"]]))
            vec = embed(text)
            vectors.append(vec)
            meta.update({"caption": caption, "tags": tags})
            rows.append(meta)
        page_token = data.get("nextPageToken")
        if not page_token: break
        time.sleep(0.05)
    return vectors, rows

# Example OneDrive ingest sketch (search by file type or folder)
def ingest_onedrive_by_query(query="jpeg", top=50):
    vectors, rows = [], []
    resp = onedrive_search(query=query, top=top)
    for it in resp.get("value", []):
        item_id = it["id"]
        web_url = it.get("webUrl")
        # Get thumbnails (choose a size)
        thumbs = onedrive_thumbnails(item_id).get("value", [])
        thumb_url = (thumbs[0].get("medium") or thumbs[0].get("small") or {}).get("url") if thumbs else None

        # Some OneDrive items may not be directly accessible anonymously; for vision,
        # prefer short-lived signed URLs via your proxy, or use provider APIs to fetch bytes if allowed.
        image_url_for_vision = thumb_url or web_url

        desc = describe_image(image_url_for_vision)
        caption = desc.get("caption","")
        tags = desc.get("tags", [])
        text = " | ".join(filter(None,[caption, ", ".join(tags), it.get('name')]))
        vec = embed(text)
        vectors.append(vec)
        rows.append({
            "source": "onedrive",
            "item_id": item_id,
            "web_url": web_url,
            "thumb_url": thumb_url,
            "taken_at": None,
            "album": None,
            "filename": it.get("name"),
            "caption": caption,
            "tags": tags
        })
    return vectors, rows

if __name__ == "__main__":
    g_vecs, g_rows = ingest_google_photos(max_pages=1)
    o_vecs, o_rows = ingest_onedrive_by_query(query="jpg", top=20)

    vectors = g_vecs + o_vecs
    rows = g_rows + o_rows

    if vectors:
        index = build_index(vectors)
        save_index_and_rows(index, rows, prefix="photos")
        print(f"Ingested {len(rows)} items total.")
    else:
        print("No vectors created.")
