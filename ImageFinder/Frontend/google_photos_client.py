# google_photos_client.py
import requests, time
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_oauth_setup import get_google_photos_credentials

BASE = "https://photoslibrary.googleapis.com/v1"

def _ensure_token(creds: Credentials):
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds.token

def list_media_items(page_size=100, page_token=None):
    creds = get_google_photos_credentials()
    token = _ensure_token(creds)
    params = {"pageSize": page_size}
    if page_token: params["pageToken"] = page_token
    r = requests.get(f"{BASE}/mediaItems", params=params, headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()

def search_media_items(filters: dict, page_size=100, page_token=None):
    creds = get_google_photos_credentials()
    token = _ensure_token(creds)
    body = {"pageSize": page_size, **({"filters": filters} if filters else {})}
    if page_token: body["pageToken"] = page_token
    r = requests.post(f"{BASE}/mediaItems:search", json=body, headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    # Quick smoke test
    data = list_media_items(page_size=5)
    print("Sample items:", [m["filename"] for m in data.get("mediaItems", [])])
