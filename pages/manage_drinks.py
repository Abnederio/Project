import streamlit as st
import pandas as pd
import os
import joblib
import subprocess
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")

# ✅ Custom CSS for a sleek UI
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }  /* Hide Sidebar */
        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            background-color: #008CBA;
            color: white;
            font-size: 18px;
            padding: 10px;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #005f7f;
            transform: scale(1.05);
        }
        div[data-testid="stDataFrame"] { 
            height: 350px !important; 
        }
    </style>
""", unsafe_allow_html=True)

# 📌 Paths
MODEL_PATH = "catboost_model.pkl"
ACCURACY_PATH = "catboost_accuracy.pkl"
IMAGE_FOLDER = "image"
DATASET_PATH = "coffee_dataset.csv"

os.makedirs(IMAGE_FOLDER, exist_ok=True)  # Ensure image folder exists

# 📥 Load dataset
df = pd.read_csv(DATASET_PATH, na_values=["None"])  

st.title("🥤 Manage Drinks")
st.markdown("### Easily Add, Edit, or Remove Coffee Menu Items")

# 📋 Show current coffee menu
st.markdown("#### ☕ Current Coffee Menu")
st.dataframe(df)

st.divider()  # Separate sections

# 🔄 Function to train and update the model
def train_and_update_model():
    st.info("🔄 Retraining the model...")

    df = pd.read_csv(DATASET_PATH, na_values=["None"])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unwanted columns

    features = ["Caffeine Level", "Sweetness", "Type", "Roast Level", "Milk Type", 
                "Flavor Notes", "Bitterness Level", "Weather"]
    target = "Coffee Name"

    df[features] = df[features].fillna("Unknown")  

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

    model = CatBoostClassifier(iterations=150, learning_rate=0.3, depth=6, verbose=0)
    model.fit(X_train, y_train, cat_features=features)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(accuracy, ACCURACY_PATH)

    st.success(f"✅ Model retrained! New accuracy: {accuracy:.2%}")

# 🎨 **Columns for Better Layout**
col1, col2, col3 = st.columns([2, 2, 1])

# ➕ **Add Coffee**
with col1:
    with st.form("add_coffee"):
        st.markdown("### ➕ Add New Coffee")

        name = st.text_input("Coffee Name", placeholder="Enter coffee name...").strip()
        caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'])
        sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'])
        drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'])
        roast_level = st.selectbox('Roast Level:', ['Medium', 'None', 'Dark'])
        milk_type = 'Dairy' if st.toggle("Do you want milk?") else 'No Dairy'
        flavor_notes = st.selectbox('Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'])
        bitterness_level = st.selectbox('Bitterness Level:', ['Low', 'Medium', 'High'])
        weather = st.selectbox('Weather:', ['Hot', 'Cold'])

        # 📸 Upload Image
        image_file = st.file_uploader("Upload an image for the coffee", type=['jpg', 'jpeg', 'png'])

        submit = st.form_submit_button("Add Coffee")

        if submit:
            if not name:
                st.error("❌ Coffee Name is required!")
            elif name in df["Coffee Name"].values:
                st.error("⚠️ Coffee already exists!")
            else:
                image_path = os.path.join(IMAGE_FOLDER, f"{name.replace(' ', '_')}.png")
                if image_file:
                    with open(image_path, "wb") as f:
                        f.write(image_file.getbuffer())
                    st.success("📸 Image uploaded successfully!")

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
                }] * 10) 

                df = pd.concat([new_entry, df], ignore_index=True)
                random_seed = np.random.randint(0, len(df) + 1)  # Seed within the range of dataset size
                df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
                df.to_csv(DATASET_PATH, index=False, na_rep="None")  
                # 🔀 Shuffle dataset with a dynamic random seed based on total rows
                
                
                # ✅ Push CSV and Image to GitHub
                subprocess.run(["git", "add", DATASET_PATH])
                subprocess.run(["git", "add", image_path])
                subprocess.run(["git", "commit", "-m", f"Updated coffee dataset and added image: {name}"])
                subprocess.run(["git", "push", "origin", "main"])

                st.success(f"☕ {name} added successfully!")
                train_and_update_model()
                st.rerun()

# ✏️ **Update Coffee**
with col2:
    st.markdown("### ✏️ Update Coffee")
    coffee_names = df["Coffee Name"].dropna().unique()
    selected_coffee = st.selectbox("Select coffee to update:", coffee_names)

    if selected_coffee:
        df["Roast Level"] = df["Roast Level"].fillna("None")
        coffee_data = df[df["Coffee Name"] == selected_coffee].iloc[0]

        new_caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Caffeine Level"]))
        new_sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Sweetness"]))
        new_drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'], index=['Frozen', 'Iced', 'Hot'].index(coffee_data["Type"]))
        new_roast_level = st.selectbox('Roast Level:', ['Medium', 'None', 'Dark'], index=['Medium', 'None', 'Dark'].index(coffee_data["Roast Level"]))
        new_flavor_notes = st.selectbox('Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'], index=['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'].index(coffee_data["Flavor Notes"]))
        new_weather = st.selectbox('Weather:', ['Hot', 'Cold'], index=['Hot', 'Cold'].index(coffee_data["Weather"]))

        if st.button("Update Coffee"):
            df.loc[df["Coffee Name"] == selected_coffee, ["Caffeine Level", "Sweetness", "Type", "Roast Level", "Flavor Notes", "Weather"]] = [new_caffeine_level, new_sweetness, new_drink_type, new_roast_level, new_flavor_notes, new_weather]

            df.to_csv(DATASET_PATH, index=False, na_rep="None")
            subprocess.run(["git", "add", DATASET_PATH])
            subprocess.run(["git", "commit", "-m", f"Updated coffee: {selected_coffee}"])
            subprocess.run(["git", "push", "origin", "main"])

            st.success(f"✅ {selected_coffee} updated successfully!")
            train_and_update_model()
            st.rerun()

# 🗑 **Delete Coffee**
with col3:
    st.markdown("### 🗑 Delete Coffee")
    delete_coffee = st.selectbox("Select coffee to delete:", df["Coffee Name"].dropna().unique())

    if st.button("Delete Coffee"):
        df = df[df["Coffee Name"] != delete_coffee]
        df.to_csv(DATASET_PATH, index=False, na_rep="None")
        
        # ✅ Remove Image from Local Storage and GitHub
        image_path = os.path.join(IMAGE_FOLDER, f"{delete_coffee.replace(' ', '_')}.png")
        if os.path.exists(image_path):
            os.remove(image_path)
            subprocess.run(["git", "rm", "--ignore-unmatch", image_path])

        # ✅ Push changes to GitHub
        subprocess.run(["git", "add", DATASET_PATH])
        subprocess.run(["git", "commit", "-m", f"Deleted coffee: {delete_coffee}"])
        subprocess.run(["git", "push", "origin", "main"])

        st.success(f"🗑 {delete_coffee} deleted successfully!")
        train_and_update_model()
        st.rerun()

st.divider()

if st.button("🏠 Go Back to Menu"):
    st.switch_page("pages/menu.py")

if st.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")


