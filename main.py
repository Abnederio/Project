import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import joblib
import os
import numpy as np
import google.generativeai as genai
import requests

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="centered")

API_URL = "https://project-a2bt.onrender.com"

# ✅ Custom CSS for a sleek UI
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }  /* Hide Sidebar */
        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            background-color: #8B4513;  /* Coffee Brown */
            color: white;
            font-size: 18px;
            padding: 10px;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #5a2e1a;
            transform: scale(1.05);
        }
        div.stAlert {
            text-align: center;
            font-size: 18px;
        }
        h1, h2, h3 {
            text-align: center;
            color: #F28C28;
        }
    </style>
""", unsafe_allow_html=True)

# 📌 Check API Status
if st.button("🔍 Check API Status"):
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        st.success("✅ API is working!")
    else:
        st.error("❌ API failed to respond.")

# 📥 Load dataset
df = pd.read_csv("coffee_dataset.csv")
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

st.divider()  # Separates sections visually

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
    
    # ✅ Fix: Remove [''] from output
    recommended_coffee = rfr_prediction[0] if isinstance(rfr_prediction, (list, np.ndarray)) else rfr_prediction  
    recommended_coffee = str(recommended_coffee).strip("[]'")  # Remove unwanted characters

    st.success(f"☕ **Your ideal coffee is: {recommended_coffee}**")

    # ✅ Fix: Check multiple image formats (.png, .jpg, .jpeg)
    image_path = None
    for ext in ["png", "jpg", "jpeg"]:
        possible_path = f"image/{recommended_coffee}.{ext}"
        if os.path.exists(possible_path):
            image_path = possible_path
            break  # Stop checking once found

    # ✅ Display Image if found
    if image_path:
        st.image(image_path, caption=f"Your coffee: {recommended_coffee}")
    else:
        st.warning("⚠️ No image available for this coffee.")

    # ✅ Gemini AI Explanation
    genai.configure(api_key="AIzaSyAXpLVdg1s1dpRj0-Crb7HYhr2xHvGUffg")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Explain why '{recommended_coffee}' was recommended based on:\n\n{features}. Explain to the end-user why it is the ideal coffee for them in only 5 sentences.")
    
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




    




