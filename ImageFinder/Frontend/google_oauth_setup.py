# google_oauth_setup.py
import os, json, pathlib
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]
TOKEN_PATH = pathlib.Path(".secrets/google_photos_token.json")
CREDS_PATH = pathlib.Path(".secrets/google_oauth_client.json")

# Create the client secret JSON file dynamically from env (or place your downloaded JSON at CREDS_PATH)
CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
if not CREDS_PATH.exists():
    client_json = {
      "installed": {
        "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
        "redirect_uris": [os.environ.get("GOOGLE_REDIRECT_URI","http://localhost:8765")],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
      }
    }
    CREDS_PATH.write_text(json.dumps(client_json, indent=2))

def get_google_photos_credentials() -> Credentials:
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDS_PATH), SCOPES
            )
            # Starts a local server to capture the redirect and exchanges code for token.
            creds = flow.run_local_server(host="localhost", port=8765, open_browser=True)
        TOKEN_PATH.write_text(creds.to_json())
    return creds

if __name__ == "__main__":
    creds = get_google_photos_credentials()
    print("Google Photos OAuth complete. Access token obtained and cached.")
