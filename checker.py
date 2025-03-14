import streamlit as st
import io
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image

# Google Drive API Authentication
CREDS_FILE = "civic-pulsar-453709-f7-10c1906e9ce5.json"
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, ["https://www.googleapis.com/auth/drive"])
drive_service = build("drive", "v3", credentials=creds)

def download_image(file_id):
    """Download an image from Google Drive and return it as a PIL image."""
    request = drive_service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% complete.")
    
    file.seek(0)  # Reset pointer
    return Image.open(file)

# File ID of the image from Google Drive
FILE_ID = "1GtQVlpBSe71mvDk5fbkICqMdUuyfyGGn"

# Streamlit UI
st.title("Google Drive Image Viewer")

st.write("Downloading and displaying the image...")

try:
    image = download_image(FILE_ID)
    st.image(image, caption="Downloaded Image", use_column_width=True)
except Exception as e:
    st.error(f"Failed to load image: {e}")




