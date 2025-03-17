import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from googleapiclient.discovery import build
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")

if 'token' not in st.session_state or st.session_state.token is None:
    st.switch_page("pages/admin.py") 

# ✅ Load Google API Credentials Securely
if "GOOGLE_CREDENTIALS" not in st.secrets:
    st.error("❌ GOOGLE_CREDENTIALS not found! Set up secrets in Streamlit Cloud.")
    st.stop()

google_creds = st.secrets["GOOGLE_CREDENTIALS"] 
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    google_creds, ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)

# ✅ Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# ✅ Google Drive Setup
FOLDER_ID = "1GNiAikLM4DAF81mrps1a6Ri2tQZGEqHi"
drive_service = build("drive", "v3", credentials=creds)


# ✅ Upload Image to Google Drive
def upload_image_to_drive(image_path, image_name):
    file_metadata = {"name": image_name, "parents": [FOLDER_ID]}
    media = {"media_body": image_path}
    
    file_drive = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    # ✅ Make file public
    drive_service.permissions().create(
        fileId=file_drive["id"],
        body={"type": "anyone", "role": "reader"}
    ).execute()

    # ✅ Return direct image link
    return f"https://drive.google.com/uc?id={file_drive['id']}"


# ✅ Retrieve Image from Google Drive
def get_image_url_from_drive(coffee_name):
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    coffee_name_formatted = coffee_name.lower().strip().replace(" ", "").replace("_", "")

    for file in files:
        file_name = file["name"].lower().strip().replace(" ", "").replace("_", "")

        if coffee_name_formatted in file_name:
            return f"https://drive.google.com/thumbnail?id={file['id']}&sz=w500"

    return None


# ✅ Load Data from Google Sheets
def load_google_sheet():
    data = sheet.get_all_records()
    return pd.DataFrame(data)


# ✅ Train & Update Model
def train_and_update_model():
    st.info("🔄 Retraining the model...")

    df = load_google_sheet()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    features = ["Caffeine Level", "Sweetness", "Type", "Roast Level", "Milk Type",
                "Flavor Notes", "Bitterness Level", "Weather"]
    target = "Coffee Name"

    df[features] = df[features].fillna("Unknown")

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

    model = CatBoostClassifier(iterations=150, learning_rate=0.3, depth=6, verbose=0)
    model.fit(X_train, y_train, cat_features=features)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, "catboost_model.pkl")
    joblib.dump(accuracy, "catboost_accuracy.pkl")

    st.success(f"✅ Model retrained! New accuracy: {accuracy:.2%}")


# ✅ Load initial data
df = load_google_sheet()

# ✅ Show Coffee Menu
st.markdown("### ☕ Current Coffee Menu")
st.dataframe(df, height=500)


# 🎯 **Add Coffee**
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
            if image_file:
                image_path = f"{name}.png"
                with open(image_path, "wb") as f:
                    f.write(image_file.getbuffer())
                image_link = upload_image_to_drive(image_path, name)
            else:
                image_link = None

            new_entry = [[name, caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level, weather, image_link]]
            sheet.append_rows(new_entry, value_input_option='RAW')

            st.success(f"✅ {name} added successfully!")
            train_and_update_model()
            st.rerun()

# 🎯 **Update Coffee**
st.markdown("### ✏️ Update Coffee")
coffee_names = df["Coffee Name"].dropna().unique()
selected_coffee = st.selectbox("Select coffee to update:", coffee_names)

if selected_coffee:
    coffee_data = df[df["Coffee Name"] == selected_coffee].iloc[0]

    # Pre-fill form with existing values
    new_name = st.text_input("Coffee Name", value=coffee_data["Coffee Name"])
    new_caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Caffeine Level"]))
    new_sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Sweetness"]))
    new_drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'], index=['Frozen', 'Iced', 'Hot'].index(coffee_data["Type"]))
    new_roast_level = st.selectbox('Roast Level:', ['Medium', 'None', 'Dark'], index=['Medium', 'None', 'Dark'].index(coffee_data["Roast Level"]))
    new_milk_type = 'Dairy' if coffee_data["Milk Type"] == "Dairy" else 'No Dairy'
    new_flavor_notes = st.selectbox('Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'], index=['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'].index(coffee_data["Flavor Notes"]))
    new_bitterness_level = st.selectbox('Bitterness Level:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Bitterness Level"]))
    new_weather = st.selectbox('Weather:', ['Hot', 'Cold'], index=['Hot', 'Cold'].index(coffee_data["Weather"]))

    image_file = st.file_uploader("Upload a new image for the coffee", type=['jpg', 'jpeg', 'png'])

    if st.button("Update Coffee"):
        df.loc[df["Coffee Name"] == selected_coffee, ["Coffee Name", "Caffeine Level", "Sweetness", "Type", "Roast Level", "Milk Type", "Flavor Notes", "Bitterness Level", "Weather"]] = [
            new_name, new_caffeine_level, new_sweetness, new_drink_type, new_roast_level, new_milk_type, new_flavor_notes, new_bitterness_level, new_weather
        ]

        if image_file:
            image_path = f"{new_name}.png"
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())
            image_link = upload_image_to_drive(image_path, new_name)
            df.loc[df["Coffee Name"] == new_name, "Image"] = image_link

        train_and_update_model()
        st.success(f"✅ {new_name} updated successfully.")
        st.rerun()
        

# 🎯 **Delete Coffee**
delete_coffee = st.selectbox("Select coffee to delete:", df["Coffee Name"].dropna().unique())
if st.button("Delete Coffee"):
    delete_coffee(delete_coffee)


st.divider()

# ✅ Sidebar Navigation
st.sidebar.markdown("### 🔧 Admin Panel")
if st.sidebar.button("🏠 Back to Menu"):
    st.switch_page("pages/menu.py")
if st.sidebar.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")










