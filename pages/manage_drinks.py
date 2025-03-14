import streamlit as st
import pandas as pd
import os
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")

# ✅ Paths
MODEL_PATH = "catboost_model.pkl"
ACCURACY_PATH = "catboost_accuracy.pkl"
DATASET_PATH = "coffee_dataset.csv"

# 🌍 Google Drive Folder ID (Replace with your Drive folder ID)
DRIVE_FOLDER_ID = "your_google_drive_folder_id"

# ✅ Authenticate Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# 📥 Load dataset safely
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH, na_values=["None"])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].astype(str).fillna("")
else:
    df = pd.DataFrame(columns=[
        "Coffee Name", "Caffeine Level", "Sweetness", "Type",
        "Roast Level", "Milk Type", "Flavor Notes", "Bitterness Level", "Weather"
    ])

st.title("🥤 Manage Drinks")
st.markdown("### Easily Add, Edit, or Remove Coffee Menu Items")

# 📋 Show current coffee menu
st.markdown("#### ☕ Current Coffee Menu")
st.dataframe(df)

st.divider()

# 🔄 Function to train and update the model
def train_and_update_model():
    st.info("🔄 Retraining the model...")

    df = pd.read_csv(DATASET_PATH, na_values=["None"])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].astype(str).fillna("")

    features = ["Caffeine Level", "Sweetness", "Type", "Roast Level", "Milk Type", 
                "Flavor Notes", "Bitterness Level", "Weather"]
    target = "Coffee Name"

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

    model = CatBoostClassifier(iterations=150, learning_rate=0.3, depth=6, verbose=0)
    model.fit(X_train, y_train, cat_features=features)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(accuracy, ACCURACY_PATH)

    st.success(f"✅ Model retrained! New accuracy: {accuracy:.2%}")

# 🌍 Upload Image to Google Drive
def upload_to_drive(file_path, file_name):
    file = drive.CreateFile({'title': file_name, 'parents': [{'id': DRIVE_FOLDER_ID}]})
    file.SetContentFile(file_path)
    file.Upload()
    return file['id']

# ➕ **Add Coffee**
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

    image_file = st.file_uploader("Upload an image for the coffee", type=['jpg', 'jpeg', 'png'])

    submit = st.form_submit_button("Add Coffee")

    if submit:
        if not name:
            st.error("❌ Coffee Name is required!")
        elif name in df["Coffee Name"].values:
            st.error("⚠️ Coffee already exists!")
        else:
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

            df = pd.concat([df, new_entry], ignore_index=True)
            df = df.sample(frac=1, random_state=None).reset_index(drop=True)

            df.to_csv(DATASET_PATH, index=False, na_rep="None")

            if image_file:
                image_path = f"/tmp/{name.replace(' ', '_')}.png"
                with open(image_path, "wb") as f:
                    f.write(image_file.getbuffer())
                
                image_id = upload_to_drive(image_path, os.path.basename(image_path))
                st.success(f"📸 Image uploaded successfully to Google Drive (ID: {image_id})")

            st.success(f"☕ {name} added successfully!")
            time.sleep(1)
            st.rerun()

# ✏️ **Update Coffee**
st.markdown("### ✏️ Update Coffee")
coffee_names = df["Coffee Name"].dropna().unique()
selected_coffee = st.selectbox("Select coffee to update:", coffee_names)

if selected_coffee:
    coffee_data = df[df["Coffee Name"] == selected_coffee].iloc[0]

    new_caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Caffeine Level"]))
    new_sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Sweetness"]))
    new_drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'], index=['Frozen', 'Iced', 'Hot'].index(coffee_data["Type"]))

    if st.button("Update Coffee"):
        df.loc[df["Coffee Name"] == selected_coffee, ["Caffeine Level", "Sweetness", "Type"]] = [new_caffeine_level, new_sweetness, new_drink_type]
        df.to_csv(DATASET_PATH, index=False, na_rep="None")

        st.success(f"✅ {selected_coffee} updated successfully!")
        time.sleep(1)
        st.rerun()

# 🗑 **Delete Coffee**
st.markdown("### 🗑 Delete Coffee")
delete_coffee = st.selectbox("Select coffee to delete:", df["Coffee Name"].dropna().unique())

if st.button("Delete Coffee"):
    df = df[df["Coffee Name"] != delete_coffee]
    df.to_csv(DATASET_PATH, index=False, na_rep="None")

    st.success(f"🗑 {delete_coffee} deleted successfully!")
    time.sleep(1)
    st.rerun()

st.divider()

if st.button("🏠 Go Back to Menu"):
    st.switch_page("pages/menu.py")

if st.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")




