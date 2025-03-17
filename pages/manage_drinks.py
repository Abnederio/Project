import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from googleapiclient.discovery import build
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")

if 'token' not in st.session_state or st.session_state.token is None:
    st.switch_page("pages/admin.py") 

# ✅ Load Google API Credentials Securely
if "GOOGLE_CREDENTIALS" not in st.secrets:
    st.error("❌ GOOGLE_CREDENTIALS not found! Set up secrets in Streamlit Cloud.")
    st.stop()

google_creds = st.secrets["GOOGLE_CREDENTIALS"] 
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    google_creds, ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)

# ✅ Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# ✅ Google Drive Setup
FOLDER_ID = "1GNiAikLM4DAF81mrps1a6Ri2tQZGEqHi"
drive_service = build("drive", "v3", credentials=creds)


# ✅ Upload Image to Google Drive
def upload_image_to_drive(image_path, image_name):
    file_metadata = {
        "name": image_name,
        "parents": [FOLDER_ID]
    }
    media = {"media_body": image_path}
    
    file_drive = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    # ✅ Make file public
    drive_service.permissions().create(
        fileId=file_drive["id"],
        body={"type": "anyone", "role": "reader"}
    ).execute()

    # ✅ Return direct image link
    return f"https://drive.google.com/uc?id={file_drive['id']}"


# ✅ Retrieve Image URL from Google Drive
def get_image_url_from_drive(coffee_name):
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    print("🔍 Retrieved Files from Google Drive:", [file["name"] for file in files])

    coffee_name_formatted = coffee_name.lower().strip().replace(" ", "").replace("_", "")

    for file in files:
        file_name = file["name"].lower().strip().replace(" ", "").replace("_", "")

        if coffee_name_formatted in file_name:  # ✅ Flexible matching
            image_url = f"https://drive.google.com/thumbnail?id={file['id']}&sz=w500"
            print(f"✅ Matched File: {file['name']} -> {image_url}")
            return image_url

    print("⚠️ No matching file found for:", coffee_name)
    return None


# ✅ Load Data from Google Sheets
def load_google_sheet():
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# ✅ Train & Update Model
def train_and_update_model():
    st.info("🔄 Retraining the model...")

    df = load_google_sheet()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    features = ["Caffeine Level", "Sweetness", "Type", "Roast Level", "Milk Type",
                "Flavor Notes", "Bitterness Level", "Weather"]
    target = "Coffee Name"

    df[features] = df[features].fillna("Unknown")

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

    model = CatBoostClassifier(iterations=150, learning_rate=0.3, depth=6, verbose=0)
    model.fit(X_train, y_train, cat_features=features)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, "catboost_model.pkl")
    joblib.dump(accuracy, "catboost_accuracy.pkl")

    st.success(f"✅ Model retrained! New accuracy: {accuracy:.2%}")


df = load_google_sheet()

# ✅ Show Coffee Menu
st.markdown("### ☕ Current Coffee Menu")
st.dataframe(df, height=500)

st.divider()

# ✅ Delete Coffee Function
def delete_coffee(delete_coffee):
    global df
    
    # Check if image exists
    coffee_data = df[df["Coffee Name"] == delete_coffee].iloc[0]
    image_link = coffee_data.get("Image", None)

    if image_link:
        image_id = image_link.split("=")[-1]  # Extract image ID
        try:
            drive_service.files().delete(fileId=image_id).execute()
            st.success(f"🗑 Image for {delete_coffee} deleted successfully from Google Drive!")
        except Exception as e:
            st.error(f"Error deleting image from Google Drive: {e}")

    # Remove from DataFrame
    
    df = df[df["Coffee Name"] != delete_coffee]

    # Remove from Google Sheets
    try:
        existing_data = sheet.get_all_records()
        rows_to_delete = [i + 2 for i, row in enumerate(existing_data) if row["Coffee Name"] == delete_coffee]

        if rows_to_delete:
            rows_to_delete.sort(reverse=True)
            for row in rows_to_delete:
                sheet.delete_rows(row)
                st.write(f"Deleted row {row} for {delete_coffee}")

            st.success(f"🗑 {delete_coffee} deleted successfully from Google Sheets!")
            train_and_update_model()
            st.rerun()
        else:
            st.error("❌ Coffee not found in Google Sheets.")

    except Exception as e:
        st.error(f"Error updating Google Sheets: {e}")

# ✅ Sidebar Navigation
st.sidebar.markdown("### 🔧 Admin Panel")
if st.sidebar.button("🏠 Back to Menu"):
    st.switch_page("pages/menu.py")
if st.sidebar.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")








