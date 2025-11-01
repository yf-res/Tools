# pip install openai requests_oauthlib faiss-cpu python-dotenv
import os, json, time, faiss, numpy as np
from openai import OpenAI
from requests_oauthlib import OAuth2Session

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-large"     # check docs for latest
VISION_MODEL = "gpt-4.1"                   # or another vision-capable model

# --- Google Photos helpers ---
def gp_list_media(oauth, page_token=None, page_size=100):
    url = "https://photoslibrary.googleapis.com/v1/mediaItems"
    params = {"pageSize": page_size}
    if page_token: params["pageToken"] = page_token
    return oauth.get(url, params=params).json()

def gp_get_oauth():
    # Use your client_id/secret + redirect, fetch tokens etc.
    # Return an OAuth2Session with token.
    ...

# --- OneDrive helpers (Graph) ---
def od_search(oauth, query):
    url = "https://graph.microsoft.com/v1.0/me/drive/root/search(q='{}')".format(query)
    return oauth.get(url).json()  # each item has webUrl, id, thumbnails (via separate call)
    # For enterprise/wide search, see Microsoft Search API.

# --- LLM captioning (vision) ---
def describe_image(image_url):
    # Use provider's ephemeral URL or a small proxy you control
    rsp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
          {"role":"system","content":"You write concise, factual photo captions and 5-10 comma-separated tags."},
          {"role":"user","content":[
              {"type":"text","text":"Caption and tags for this photo. Output JSON with {caption, tags}."},
              {"type":"image_url","image_url":{"url": image_url}}
          ]}
        ],
        temperature=0.2,
    )
    txt = rsp.choices[0].message.content
    try:
        return json.loads(txt)
    except:
        # fall back: strip code fences, re-ask, etc.
        return {"caption": txt, "tags": []}

# --- Embedding utility ---
def embed(text):
    er = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(er.data[0].embedding, dtype="float32")

# --- Indexer (batch) ---
def index_google_photos():
    gp = gp_get_oauth()
    token = None
    vectors, rows = [], []
    while True:
        data = gp_list_media(gp, token)
        for m in data.get("mediaItems", []):
            base_url = m["baseUrl"]  # add size params when fetching thumbs
            web_url  = m.get("productUrl", base_url)
            meta = {
                "source": "google",
                "item_id": m["id"],
                "web_url": web_url,
                "thumb_url": base_url,
                "taken_at": m.get("mediaMetadata", {}).get("creationTime"),
                "album": None,  # augment later if you map albums
            }
            desc = describe_image(base_url)
            caption = desc.get("caption","")
            tags = desc.get("tags",[])
            text_for_embed = " | ".join(filter(None, [
                caption, ", ".join(tags), m.get("filename","")
            ]))
            vec = embed(text_for_embed)
            vectors.append(vec)
            meta.update({"caption": caption, "tags": tags})
            rows.append(meta)
        token = data.get("nextPageToken")
        if not token: break
        time.sleep(0.1)

    # Build FAISS index
    if vectors:
        dim = len(vectors[0])
        index = faiss.IndexFlatIP(dim)
        X = np.vstack(vectors)
        faiss.normalize_L2(X)
        index.add(X)
        return index, rows
    return None, []

# --- Search runtime ---
def search(query, index, rows, k=20):
    qv = embed(query).reshape(1, -1)
    faiss.normalize_L2(qv)
    scores, idxs = index.search(qv, k)
    results = []
    for i,score in zip(idxs[0], scores[0]):
        meta = rows[i].copy()
        meta["score"] = float(score)
        results.append(meta)
    return results
