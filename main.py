import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import google.generativeai as genai
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from googleapiclient.discovery import build
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(initial_sidebar_state="expanded", page_title="Coffee Recommender", layout="centered")

# ✅ Sidebar Styling
st.markdown(
    """
    <style>
        body {
            background-color: #A27B5C;
        }
        .stApp {
            background-color: #A27B5C;
        }
        .recommend-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .recommend-button {
            background-color: #8D6C4F;
            color: white;
            padding: 14px 28px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: 0.3s ease-in-out;
            margin-top: 20px;
        }
        .recommend-button:hover {
            background-color: #754F44;
            transform: scale(1.05);
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #6B4E3D;
            color: white;
        }
        .sidebar-title {
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sidebar-button {
            display: flex;
            justify-content: center;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "civic-pulsar-453709-f7-10c1906e9ce5.json"

# ✅ Authenticate Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# ✅ Google Drive Setup (For Image Retrieval)
FOLDER_ID = "1GtQVlpBSe71mvDk5fbkICqMdUuyfyGGn"
drive_service = build("drive", "v3", credentials=creds)

def load_google_sheet():
    """Load coffee data from Google Sheets."""
    data = sheet.get_all_records()
    return pd.DataFrame(data)

df = load_google_sheet()

# ✅ Prepare Dataset for Model
X = df.drop(columns=['Coffee Name'])
y = df['Coffee Name']

X.fillna("Unknown", inplace=True)
y.fillna("Unknown", inplace=True)

cat_features = list(range(X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

MODEL_PATH = "catboost_model.pkl"
ACCURACY_PATH = "catboost_accuracy.pkl"

# ✅ Load or Train Model
if os.path.exists(MODEL_PATH) and os.path.exists(ACCURACY_PATH):
    model = joblib.load(MODEL_PATH)
    accuracy = joblib.load(ACCURACY_PATH)
else:
    model = CatBoostClassifier(iterations=150, learning_rate=0.3, depth=6, verbose=0)
    model.fit(X_train, y_train, cat_features=cat_features)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(accuracy, ACCURACY_PATH)

st.image("Header.png", width=800)
st.header("☕ Alex's Coffee Haven: AI Coffee Recommender")
st.divider()

# ✅ Retrieve Image from Google Drive
def get_image_url_from_drive(coffee_name):
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    coffee_name_formatted = coffee_name.lower().replace(" ", "").replace("_", "")

    for file in files:
        file_name = file['name'].lower().replace(" ", "").replace("_", "")
        if file_name.startswith(coffee_name_formatted) and file_name.endswith(('.png', '.jpg', '.jpeg')):
            return f"https://drive.google.com/thumbnail?id={file['id']}&sz=w500"

    return None

# 🎯 **User Input Section**
st.markdown("#### ☕ Select Your Preferences")

col1, col2 = st.columns(2)

with col1:
    caffeine_level = st.selectbox('☕ Caffeine Level:', ['Low', 'Medium', 'High'])
    sweetness = st.selectbox('🍬 Sweetness:', ['Low', 'Medium', 'High'])
    drink_type = st.selectbox('❄️ Drink Type:', ['Frozen', 'Iced', 'Hot'])
    roast_level = st.selectbox('🔥 Roast Level:', ['Medium', 'None', 'Dark'])

with col2:
    milk_type = 'Dairy' if st.toggle("🥛 Do you want milk?") else 'No Dairy'
    flavor_notes = st.selectbox('🍫 Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'])
    bitterness_level = st.selectbox('🏴 Bitterness Level:', ['Low', 'Medium', 'High'])
    weather = st.selectbox('🌡 Weather:', ['Hot', 'Cold'])

st.divider()

# 🌟 **Recommendation Section**
st.markdown('<div class="recommend-container">', unsafe_allow_html=True)
st.markdown("### ☕ AI Coffee Recommendation")

if st.button("🎯 Recommend Coffee", key="recommend"):
    rfr_input_data = [[caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level, weather]]
    rfr_prediction = model.predict(rfr_input_data)
    recommended_coffee = str(rfr_prediction[0]).strip("[]'")

    st.success(f"☕ **Your ideal coffee is: {recommended_coffee}**")

    # ✅ Get Image from Google Drive
    image_link = get_image_url_from_drive(recommended_coffee)

    if image_link:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <img src="{image_link}" width="400" style="border-radius: 12px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ No image available for this coffee.")

    # ✅ AI Explanation
    prompt = f"""
    Explain why '{recommended_coffee}' is the best match for the user based on:
    - Caffeine Level: {caffeine_level}
    - Sweetness: {sweetness}
    - Drink Type: {drink_type}
    - Roast Level: {roast_level}
    - Milk Type: {milk_type}
    - Flavor Notes: {flavor_notes}
    - Bitterness Level: {bitterness_level}
    - Weather: {weather}
    """

    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    explanation = response.text if response else "No explanation available."

    st.markdown("#### 💡 Why this coffee?")
    st.info(explanation)

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ✅ Sidebar Admin Button
with st.sidebar:
    st.markdown('<p class="sidebar-title">🔑 Admin Access</p>', unsafe_allow_html=True)
    if st.button("Go to Admin Dashboard"):
        st.switch_page("pages/admin.py")









    




