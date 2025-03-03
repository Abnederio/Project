import streamlit as st
import pandas as pd
import os
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")  # ✅ Move this to the top

# ✅ Apply custom CSS for a wider main content area
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
    </style>
""", unsafe_allow_html=True)

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

# For Columns
col1, col2, col3 = st.columns([2,2,1])

with col1:
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
                }] * 5) 

                df = pd.concat([new_entry, df], ignore_index=True)
                
                # 🔀 Shuffle dataset with a dynamic random seed based on total rows
                random_seed = np.random.randint(0, len(df) + 1)  # Seed within the range of dataset size
                df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

                # ✅ Save with "None" explicitly stored
                df.to_csv(DATASET_PATH, index=False, na_rep="None")  

                st.success(f"☕ {name} added successfully!")

                # 🔄 Retrain model with new data
                train_and_update_model()

                st.rerun()

# ✏️ Update Existing Coffee
with col2:
    st.write("### ✏️ Update Coffee")
    coffee_names = df["Coffee Name"].dropna().unique()  # Remove NaN values
    selected_coffee = st.selectbox("Select coffee to update:", coffee_names)

    if selected_coffee:
        coffee_data = df[df["Coffee Name"] == selected_coffee].iloc[0]

        # Function to handle NaN values safely
        def get_valid_index(value, options):
            if pd.isna(value) or value not in options:  # Check if NaN or invalid
                return 0  # Default to the first valid option
            return options.index(value)

        # Fixing the issue for all fields
        new_caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'], 
                                        index=get_valid_index(coffee_data["Caffeine Level"], ['Low', 'Medium', 'High']))
        new_sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'], 
                                    index=get_valid_index(coffee_data["Sweetness"], ['Low', 'Medium', 'High']))
        new_drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'], 
                                    index=get_valid_index(coffee_data["Type"], ['Frozen', 'Iced', 'Hot']))
        new_roast_level = st.selectbox('Roast Level:', ['Medium', 'None', 'Dark'], 
                                    index=get_valid_index(coffee_data["Roast Level"], ['Medium', 'None', 'Dark']))
        new_milk_type = st.selectbox('Milk Type:', ['Dairy', 'No Dairy'], 
                                    index=get_valid_index(coffee_data["Milk Type"], ['Dairy', 'No Dairy']))
        new_flavor_notes = st.selectbox('Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'], 
                                        index=get_valid_index(coffee_data["Flavor Notes"], ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso']))
        new_bitterness_level = st.selectbox('Bitterness Level:', ['Low', 'Medium', 'High'], 
                                            index=get_valid_index(coffee_data["Bitterness Level"], ['Low', 'Medium', 'High']))
        new_weather = st.selectbox('Weather:', ['Hot', 'Cold'], 
                                index=get_valid_index(coffee_data["Weather"], ['Hot', 'Cold']))

        if st.button("Update Coffee"):
            df.loc[df["Coffee Name"] == selected_coffee, ["Caffeine Level", "Sweetness", "Type", "Roast Level", "Milk Type", "Flavor Notes", "Bitterness Level", "Weather"]] = [
                new_caffeine_level, new_sweetness, new_drink_type, new_roast_level, new_milk_type, new_flavor_notes, new_bitterness_level, new_weather]

            df.to_csv(DATASET_PATH, index=False, na_rep="None")
            st.success(f"✅ {selected_coffee} updated successfully!")
            train_and_update_model()
            st.rerun()

# 🗑 Delete Coffee
with col3:
    st.write("### 🗑 Delete Coffee")

    # Remove NaN coffee names before displaying
    valid_coffee_names = df["Coffee Name"].dropna().unique()
    delete_coffee = st.selectbox("Select coffee to delete:", valid_coffee_names)

    if st.button("Delete Coffee"):
        if delete_coffee in df["Coffee Name"].values:
            # Remove the selected coffee from dataset
            df = df[df["Coffee Name"] != delete_coffee]

            # Delete associated image
            image_path = os.path.join(IMAGE_FOLDER, f"{delete_coffee.replace(' ', '_')}.png")
            if os.path.exists(image_path):
                os.remove(image_path)  # ✅ Delete the image file
                st.success("🖼 Image deleted successfully!")

            # Save updated dataset
            df.to_csv(DATASET_PATH, index=False, na_rep="None")

            st.success(f"🗑 {delete_coffee} deleted successfully!")
            
            # Retrain model after deletion
            train_and_update_model()
            
            st.rerun()
        else:
            st.warning("⚠️ Selected coffee does not exist in the dataset.")
            
if st.button("🏠 Go Back to Menu"):
    st.switch_page("pages/menu.py")
if st.button("🚪 Logout"):
        st.session_state.token = None  # Clear token
        st.switch_page("pages/admin.py")

