import os
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
flow = InstalledAppFlow.from_client_secrets_file(
    os.environ.get('GMAIL_OAUTH_CREDENTIALS', 'client_secret.json'), SCOPES)
creds = flow.run_local_server(port=0)
with open('gmail_oauth_token.json', 'w') as f:
    f.write(creds.to_json())
print('Saved gmail_oauth_token.json')
