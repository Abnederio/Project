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

# ✅ Custom CSS for a Premium Coffee-Themed UI
st.markdown(
    """
    <style>
        /* Background */
        body {
            background-color: #F8EDE3 !important;  /* Soft Latte */
        }
        .stApp {
            background-color: #F8EDE3 !important;  /* Warm and inviting */
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #6D4C3D !important;  /* Deep Mocha */
            color: white !important;
        }

        /* Buttons */
        div.stButton > button {
            width: 100%;
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            transition: 0.3s ease-in-out;
            border: none;
            font-weight: bold;
        }

        /* Caramel Button */
        div.stButton > button:first-child {
            background-color: #A67B5B;  /* Caramel */
            color: white;
        }
        div.stButton > button:first-child:hover {
            background-color: #8D6C4F;  /* Toasted Brown */
            transform: scale(1.05);
        }

        /* Mocha Button (Admin Login) */
        div.stButton > button:last-child {
            background-color: #5E503F;  /* Dark Cocoa */
            color: white;
        }
        div.stButton > button:last-child:hover {
            background-color: #4A4032;  /* Richer Mocha */
            transform: scale(1.05);
        }

        /* Headers & Text */
        h1, h2, h3, h4, h5, h6 {
            color: #3E2723 !important;  /* Roasted Coffee */
        }
        p, div {
            color: #4E342E !important;  /* Dark Cocoa Text */
        }

        /* Dataset Table */
        .stDataFrame {
            background-color: #F0D9B5 !important;  /* Light Cappuccino */
            color: #3E2723 !important;  /* Mocha text */
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

# ✅ Google Drive Setup
FOLDER_ID = "1GtQVlpBSe71mvDk5fbkICqMdUuyfyGGn"
drive_service = build("drive", "v3", credentials=creds)

def load_google_sheet():
    """Load coffee data from Google Sheets."""
    data = sheet.get_all_records()
    return pd.DataFrame(data)

df = load_google_sheet()

# ✅ Machine Learning Setup
X = df.drop(columns=['Coffee Name'])
y = df['Coffee Name']

X.fillna("Unknown", inplace=True)
y.fillna("Unknown", inplace=True)

cat_features = list(range(X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

MODEL_PATH = "catboost_model.pkl"
ACCURACY_PATH = "catboost_accuracy.pkl"

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

st.markdown(f"**✅ Model Accuracy:** `{accuracy:.2%}`")

st.image("Header.png", width=700)
st.header("☕ Alex's Coffee Haven: AI Coffee Recommender")
st.divider()

# ✅ Sidebar Navigation + Admin Login
with st.sidebar:
    st.title("☕ Navigation")
    
    st.markdown("🔑 **Admin Access**")
    if st.button("👨‍💼 Admin Login"):
        st.switch_page("pages/admin.py")

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
st.markdown("### ☕ AI Coffee Recommendation")

if st.button("🎯 Recommend Coffee"):
    rfr_prediction = model.predict([[caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level, weather]])
    recommended_coffee = str(rfr_prediction[0]).strip("[]'")

    st.success(f"☕ **Your ideal coffee is: {recommended_coffee}**")

    image_link = get_image_url_from_drive(recommended_coffee)
    if image_link:
        st.image(image_link, caption=f"Your coffee: {recommended_coffee}")
    else:
        st.warning("⚠️ No image available for this coffee.")

st.divider()








    




