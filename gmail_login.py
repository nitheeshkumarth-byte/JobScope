import os
from pathlib import Path
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    client = root / (os.environ.get("GMAIL_OAUTH_CREDENTIALS") or "client_secret.json")
    token = root / (os.environ.get("GMAIL_OAUTH_TOKEN_FILE") or "gmail_oauth_token.json")
    scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    flow = InstalledAppFlow.from_client_secrets_file(str(client), scopes)
    creds = flow.run_local_server(port=0)
    token.write_text(creds.to_json(), encoding="utf-8")
    print("Saved the Gmail connection for JobScope.")
