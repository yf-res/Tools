# ms_oauth_setup.py
import os, json, pathlib, time
from dotenv import load_dotenv
import msal
import requests

load_dotenv()

MS_CLIENT_ID = os.environ["MS_CLIENT_ID"]
MS_TENANT_ID = os.environ.get("MS_TENANT_ID", "consumers")  # or your tenant GUID
AUTHORITY = f"https://login.microsoftonline.com/{MS_TENANT_ID}"
SCOPES = ["Files.Read", "offline_access"]  # add more if needed

TOKEN_CACHE_PATH = pathlib.Path(".secrets/msal_token_cache.json")
TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_cache():
    cache = msal.SerializableTokenCache()
    if TOKEN_CACHE_PATH.exists():
        cache.deserialize(TOKEN_CACHE_PATH.read_text())
    return cache

def _save_cache(cache):
    TOKEN_CACHE_PATH.write_text(cache.serialize())

def get_graph_access_token() -> str:
    cache = _load_cache()
    app = msal.PublicClientApplication(client_id=MS_CLIENT_ID, authority=AUTHORITY, token_cache=cache)

    # Try silent
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            _save_cache(cache); return result["access_token"]

    # Device code flow
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError("Failed to create device flow.")
    print(f"\nTo sign in, visit: {flow['verification_uri']}\nEnter the code: {flow['user_code']}\n")
    result = app.acquire_token_by_device_flow(flow)  # blocks until done or timeout
    if "access_token" in result:
        _save_cache(cache); return result["access_token"]
    raise RuntimeError(f"Auth failed: {result.get('error_description')}")

# Example OneDrive calls
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

def onedrive_me_root_children(top=5):
    token = get_graph_access_token()
    r = requests.get(f"{GRAPH_BASE}/me/drive/root/children?$top={top}",
                     headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()

def onedrive_search(query, top=10):
    token = get_graph_access_token()
    r = requests.get(f"{GRAPH_BASE}/me/drive/root/search(q='{query}')?$top={top}",
                     headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()

def onedrive_thumbnails(item_id):
    token = get_graph_access_token()
    r = requests.get(f"{GRAPH_BASE}/me/drive/items/{item_id}/thumbnails",
                     headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    print("Listing OneDrive root children:")
    data = onedrive_me_root_children()
    for it in data.get("value", []):
        print("-", it.get("name"), it.get("webUrl"))
