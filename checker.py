from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

FOLDER_ID = "1GtQVlpBSe71mvDk5fbkICqMdUuyfyGGn"
CREDS_FILE = "civic-pulsar-453709-f7-10c1906e9ce5.json"

# Authenticate with Google Drive
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, ["https://www.googleapis.com/auth/drive"])
drive_service = build("drive", "v3", credentials=creds)

def list_files():
    """Test if the service account can access the Google Drive folder."""
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        print("✅ Service account can access the folder. Here are the files:")
        for file in files:
            print(f"- {file['name']} (ID: {file['id']})")
    else:
        print("❌ No files found OR service account has no access.")

list_files()



