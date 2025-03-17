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
import json
import requests

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="centered")
# CSS
st.markdown(
    """
    <style>
    body {
        background-color: #A27B5C;
    }
    .stApp {
        background-color: #A27B5C; 
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ✅ Load Google API Credentials Securely (from Streamlit Secrets)
if "GOOGLE_CREDENTIALS" not in st.secrets:
    st.error("❌ GOOGLE_CREDENTIALS not found! Set up secrets in Streamlit Cloud.")
    st.stop()

google_creds = st.secrets["GOOGLE_CREDENTIALS"] 
print(st.secrets["GOOGLE_CREDENTIALS"]["client_email"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    google_creds, ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)

# ✅ Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# ✅ Google Drive Setup (For Image Retrieval)
FOLDER_ID = "1GNiAikLM4DAF81mrps1a6Ri2tQZGEqHi"
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

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

st.image("Header.png", width=1000)
st.header("☕ Alex's Coffee Haven: AI Coffee Recommender")
st.divider()

# ✅ Retrieve Image from Google Drive

# ✅ Retrieve Image from Google Drive
def get_image_url_from_drive(coffee_name):
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    print("🔍 Retrieved Files from Google Drive:", [file['name'] for file in files])

    coffee_name_formatted = coffee_name.lower().strip().replace(" ", "").replace("_", "")

    for file in files:
        file_name = file['name'].lower().strip().replace(" ", "").replace("_", "")
        
        if coffee_name_formatted in file_name:  # ✅ Looser match
            image_url = f"https://drive.google.com/thumbnail?id={file['id']}&sz=w500"
            print(f"✅ Matched File: {file['name']} -> {image_url}")
            return image_url

    print("⚠️ No matching file found for:", coffee_name)
    return None


# 🎯 **User Input Section**
st.markdown("#### ☕ Select Your Preferences")

# Function to style placeholder text
def format_placeholder(option):
    return "Select an option" if option == "Select an option" else option

# 🏗 **Columns for Better Layout**
col1, col2 = st.columns(2)

with col1:
    caffeine_level = st.selectbox('☕ Caffeine Level:', ['Select an option', 'Low', 'Medium', 'High'], format_func=format_placeholder)
    sweetness = st.selectbox('🍬 Sweetness:', ['Select an option', 'Low', 'Medium', 'High'], format_func=format_placeholder)
    drink_type = st.selectbox('❄️ Drink Type:', ['Select an option', 'Frozen', 'Iced', 'Hot'], format_func=format_placeholder)
    roast_level = st.selectbox('🔥 Roast Level:', ['Select an option', 'Medium', 'None', 'Dark'], format_func=format_placeholder)

with col2:
    milk_type = 'Dairy' if st.toggle("🥛 Do you want milk?") else 'No Dairy'
    flavor_notes = st.selectbox('🍫 Flavor Notes:', ['Select an option', 'Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'], format_func=format_placeholder)
    bitterness_level = st.selectbox('🏴 Bitterness Level:', ['Select an option', 'Low', 'Medium', 'High'], format_func=format_placeholder)
    weather = st.selectbox('🌡 Weather:', ['Select an option', 'Hot', 'Cold'], format_func=format_placeholder)

st.divider()

# Custom CSS to make placeholder text gray
st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div {
        color: gray !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 🌟 **Recommendation Section**
st.markdown("### ☕ AI Coffee Recommendation")

if "recommended_coffee" not in st.session_state:
    st.session_state.recommended_coffee = None

# **Formatted Feature String**
features = f"""
- ☕ Caffeine Level: `{caffeine_level}`
- 🍬 Sweetness: `{sweetness}`
- ❄️ Drink Type: `{drink_type}`
- 🔥 Roast Level: `{roast_level}`
- 🥛 Milk Type: `{milk_type}`
- 🍫 Flavor Notes: `{flavor_notes}`
- 🏴 Bitterness Level: `{bitterness_level}`
- 🌡 Weather: `{weather}`
"""

if st.button("🎯 Recommend Coffee"):
    
    if "Select an option" in [caffeine_level, sweetness, drink_type, roast_level, flavor_notes, bitterness_level, weather]:
        st.error("❌ Please fill up all of the options for better recommendation!") 
        
    else:
        
        rfr_input_data = [[caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level, weather]]
        rfr_prediction = model.predict(rfr_input_data)
        
        recommended_coffee = rfr_prediction[0] if isinstance(rfr_prediction, (list, np.ndarray)) else rfr_prediction  
        recommended_coffee = str(recommended_coffee).strip("[]'")  

        st.success(f"☕ **Your ideal coffee is: {recommended_coffee}**")

        # ✅ Get Image from Google Drive
        image_link = get_image_url_from_drive(recommended_coffee)

        if image_link:
            response = requests.get(image_link)

            # Check if response is an image
            if "image" not in response.headers.get("Content-Type", ""):
                st.error("🚨 Error: The image is not valid! Google Drive may be blocking access.")
                st.write(f"[View Image in Drive]({image_link})")
            else:
                st.image(response.content, width=500)
        else:
            print("Image URL:", image_link)
            st.warning("⚠️ No image available for this coffee.")

        # ✅ Gemini AI Explanation
        genai.configure(api_key="AIzaSyAXpLVdg1s1dpRj0-Crb7HYhr2xHvGUffg")
        ai_model = genai.GenerativeModel("gemini-2.0-flash")
        response2 = ai_model.generate_content(f"Explain why '{recommended_coffee}' was recommended based on:\n\n{features} make it like a true salesperson. Explain in 5 sentences.")
        
        explanation = response2.text

        if explanation:
            st.markdown(f"#### 💡 Why this coffee?")
            st.info(explanation)
        else:
            st.warning("🤖 AI couldn't generate an explanation. Please try again.")

st.divider()

# ✅ Sidebar Admin Button
with st.sidebar:
    st.markdown('<p class="sidebar-title">🔑 Admin Access</p>', unsafe_allow_html=True)
    if st.button("Admin Login"):
        st.switch_page("pages/admin.py")








    




