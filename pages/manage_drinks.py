import streamlit as st
import pandas as pd
import os
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 📌 Paths
MODEL_PATH = "catboost_model.pkl"
ACCURACY_PATH = "catboost_accuracy.pkl"
IMAGE_FOLDER = "images"
DATASET_PATH = "coffee_dataset.csv"

# 📂 Ensure the image folder exists
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# 📥 Load dataset
df = pd.read_csv(DATASET_PATH, na_values=["None"])  # Ensure "None" is read properly

st.title("🥤 Manage Drinks")

# 📋 Show current coffee menu
st.write("### ☕ Current Coffee Menu")
st.dataframe(df)

# 🔄 Function to train and update the model
def train_and_update_model():
    st.info("🔄 Retraining the model...")

    # Load dataset
    df = pd.read_csv(DATASET_PATH, na_values=["None"])  

    # Drop any unnamed columns (e.g., "Unnamed: 9")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Define features and target
    features = ["Caffeine Level", "Sweetness", "Type", "Roast Level", "Milk Type", 
                "Flavor Notes", "Bitterness Level", "Weather"]
    target = "Coffee Name"

    # ✅ Replace NaN values with "Unknown" (Same as main.py)
    df[features] = df[features].fillna("Unknown")  

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

    # Train new model
    model = CatBoostClassifier(iterations=150, learning_rate=0.3, depth=6, task_type="GPU", verbose=0)
    model.fit(X_train, y_train, cat_features=features)

    # Evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model & accuracy
    joblib.dump(model, MODEL_PATH)
    joblib.dump(accuracy, ACCURACY_PATH)

    st.success(f"✅ Model retrained! New accuracy: {accuracy:.2%}")

# ➕ Add new coffee
with st.form("add_coffee"):
    st.write("### ➕ Add New Coffee")
    name = st.text_input("Coffee Name")
    caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'])
    sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'])
    drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'])
    roast_level = st.selectbox('Roast Level:', ['Medium', 'None', 'Dark'])
    milk_type = 'Dairy' if st.toggle("Do you want milk?") else 'No Dairy'
    flavor_notes = st.selectbox('Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'])
    bitterness_level = st.selectbox('Bitterness Level:', ['Low', 'Medium', 'High'])
    weather = st.selectbox('Weather:', ['Hot', 'Cold'])

    # 📸 Upload image
    image_file = st.file_uploader("Upload an image for the coffee", type=['jpg', 'jpeg', 'png'])

    submit = st.form_submit_button("Add Coffee")

    if submit:
        if not name:
            st.error("❌ Coffee Name is required!")
        else:
            # Save image if uploaded
            image_path = os.path.join(IMAGE_FOLDER, f"{name.replace(' ', '_')}.png")
            if image_file:
                with open(image_path, "wb") as f:
                    f.write(image_file.getbuffer())
                st.success("📸 Image uploaded successfully!")

            # Add new entry to the dataset
            new_entry = pd.DataFrame([{
                "Coffee Name": name,
                "Caffeine Level": caffeine_level,
                "Sweetness": sweetness,
                "Type": drink_type,
                "Roast Level": roast_level,
                "Milk Type": milk_type,
                "Flavor Notes": flavor_notes,
                "Bitterness Level": bitterness_level,
                "Weather": weather,
            }])

            df = pd.concat([new_entry, df], ignore_index=True)

            # ✅ Save with "None" explicitly stored
            df.to_csv(DATASET_PATH, index=False, na_rep="None")  

            st.success(f"☕ {name} added successfully!")

            # 🔄 Retrain model with new data
            train_and_update_model()

            st.rerun()

