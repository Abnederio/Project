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

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="centered")

# ✅ Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "civic-pulsar-453709-f7-10c1906e9ce5.json"  # Ensure this file is in the project directory

# ✅ Authenticate Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# ✅ Google Drive Setup (For Image Retrieval)
FOLDER_ID = "1GtQVlpBSe71mvDk5fbkICqMdUuyfyGGn"

# Authenticate with Google Drive API
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

st.markdown(f"**✅ Model Accuracy:** `{accuracy:.2%}`")

st.header("☕ Alex's Coffee Haven: AI Coffee Recommender")
st.divider()

# ✅ Retrieve Image from Google Drive
def get_image_url_from_drive(coffee_name):
    """Search for a matching image in Google Drive and return a direct link."""
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    coffee_name_formatted = coffee_name.lower().replace(" ", "").replace("_", "")

    for file in files:
        file_name = file['name'].lower().replace(" ", "").replace("_", "")

        if file_name.startswith(coffee_name_formatted) and file_name.endswith(('.png', '.jpg', '.jpeg')):
            return f"https://drive.google.com/uc?id={file['id']}"

    return None  # No matching image found

# 🎯 **User Input Section**
st.markdown("#### ☕ Select Your Preferences")

# 🏗 **Columns for Better Layout**
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
    rfr_input_data = [[caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level, weather]]
    rfr_prediction = model.predict(rfr_input_data)
    
    recommended_coffee = rfr_prediction[0] if isinstance(rfr_prediction, (list, np.ndarray)) else rfr_prediction  
    recommended_coffee = str(recommended_coffee).strip("[]'")  

    st.success(f"☕ **Your ideal coffee is: {recommended_coffee}**")

    # ✅ Get Image from Google Drive
    image_link = get_image_url_from_drive(recommended_coffee)

    if image_link:
        print(image_link)
        st.image(image_link, caption=f"Your coffee: {recommended_coffee}")
    else:
        st.warning("⚠️ No image available for this coffee.")

    # ✅ Gemini AI Explanation
    genai.configure(api_key="YOUR_GEMINI_API_KEY")
    ai_model = genai.GenerativeModel("gemini-2.0-flash")
    response = ai_model.generate_content(f"Explain why '{recommended_coffee}' was recommended based on:\n\n{features}. Explain in 5 sentences.")
    
    explanation = response.text

    if explanation:
        st.markdown(f"#### 💡 Why this coffee?")
        st.info(explanation)
    else:
        st.warning("🤖 AI couldn't generate an explanation. Please try again.")

st.divider()

# ✅ Admin Button
if st.button("🔑 Admin Login"):
    st.switch_page("pages/admin.py")








    




