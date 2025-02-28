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

# Set up Gemini API
genai.configure(api_key="AIzaSyAXpLVdg1s1dpRj0-Crb7HYhr2xHvGUffg")
# Gemini Explanation Function
def get_explanation(recommended_coffee, features):  
    model = genai.GenerativeModel("gemini-2.0-flash")  
    response = model.generate_content(f"Explain why '{recommended_coffee}' was recommended based on:\n\n{features}. Explain to the end-user why is it ideal coffee for her/him. Make it only 5 sentences.")
    return response.text  # Extract explanation text

# Admin Button
if st.sidebar.button("🔑 Admin Login"):
    st.query_params["page"] = "admin"
    st.rerun()

# Get current query parameters
query_params = st.query_params

# Check if the page is "admin"
if query_params.get("page") == "admin":
    import admin  # Load `admin.py`

# Load dataset
df = pd.read_csv("coffee_dataset.csv")

X = df.drop(columns=['Coffee Name'])
y = df['Coffee Name']

X.fillna("Unknown", inplace=True)
y.fillna("Unknown", inplace=True)

cat_features = list(range(X.shape[1]))  # Ensure correct categorical indices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MODEL_PATH = "catboost_model.pkl"
ACCURACY_PATH = "catboost_accuracy.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(ACCURACY_PATH):
    model = joblib.load(MODEL_PATH)
    accuracy = joblib.load(ACCURACY_PATH)
else:
    model = CatBoostClassifier(iterations=100, learning_rate=0.2, depth=9, verbose=0)
    model.fit(X_train, y_train, cat_features=cat_features)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(accuracy, ACCURACY_PATH)

st.write(f"**Model Accuracy:** {accuracy:.2f}%")

st.header(" Alex's Coffee Haven Coffee Recommender ☕")

# Input boxes for features
caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'])
sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'])
drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'])
roast_level = st.selectbox('Roast Level:', ['Medium', 'None', 'Dark'])
milk_type = 'Dairy' if st.toggle("Do you want milk?") else 'None'
flavor_notes = st.selectbox('Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'])
bitterness_level = st.selectbox('Bitterness Level:', ['Low', 'Medium', 'High'])
weather = st.selectbox('Weather:', ['Hot', 'Cold'])

# Ensure recommended_coffee exists in session state
if "recommended_coffee" not in st.session_state:
    st.session_state.recommended_coffee = None

# Store features as a string
features = f"""
- Caffeine Level: {caffeine_level}
- Sweetness: {sweetness}
- Drink Type: {drink_type}
- Roast Level: {roast_level}
- Milk Type: {milk_type}
- Flavor Notes: {flavor_notes}
- Bitterness Level: {bitterness_level}
- Weather: {weather}
"""

# Recommendation Button
if st.button('Recommend', key='rfr_detect'):
    # Predict the recommended coffee
    rfr_input_data = [[caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level, weather]]
    rfr_prediction = model.predict(rfr_input_data)

    recommended_coffee = rfr_prediction[0]

    # Ensure it's a plain string
    if isinstance(recommended_coffee, (list, np.ndarray)):  
        recommended_coffee = recommended_coffee[0]

    recommended_coffee = str(recommended_coffee)  

    # Store the recommendation in session state
    st.session_state.recommended_coffee = recommended_coffee

    # Display the recommendation
    st.success(f"☕ The coffee we recommend is: **{recommended_coffee}**")

    # Display the image
    image_path = f"images/{recommended_coffee}.png"
    if os.path.exists(image_path):
        st.image(image_path, caption=f"{recommended_coffee}", use_column_width=True)
    else:
        st.warning("Image not available for this coffee.")

    if st.session_state.recommended_coffee:  # Check if a coffee was recommended
        explanation = get_explanation(st.session_state.recommended_coffee, features)
        st.write(f"*Why this coffee?*\n\n{explanation}")
    else:
        st.warning("Please click 'Recommend' first to get a coffee suggestion.")
    




