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
from googleapiclient.http import MediaFileUpload
import random

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")

if 'token' not in st.session_state or st.session_state.token is None:
    st.switch_page("pages/admin.py") 
    
st.markdown(
    """
    <style>
        /* Background */
        body {
            background-color: #A27B5C;  /* Warm Coffee Tone */
        }
        .stApp {
            background-color: #A27B5C; 
        }

        /* Submit Button */
        div.stButton > button:last-child {
            background-color: #3E2723;  /* Espresso Brown */
            color: #FFFFFF;  /* White Text */
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            border: 2px solid #5D4037;  /* Subtle Border */
            transition: all 0.3s ease-in-out;
        }

        /* Hover Effect */
        div.stButton > button:last-child:hover {
            background-color: #4E342E;  /* Richer Coffee */
            transform: scale(1.08);
            border-color: #3E2723;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }

        /* Click Effect */
        div.stButton > button:last-child:active {
            transform: scale(0.95);
            background-color: #2E1B14;  /* Strong Espresso */
        }
    </style>
    """,
    unsafe_allow_html=True
)


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

    # ✅ Use MediaFileUpload instead of a dictionary
    media = MediaFileUpload(image_path, mimetype="image/png")  

    file_drive = drive_service.files().create(
        body=file_metadata,
        media_body=media,  # ✅ Corrected here
        fields="id"
    ).execute()

    # ✅ Make file public
    drive_service.permissions().create(
        fileId=file_drive["id"],
        body={"type": "anyone", "role": "reader"}
    ).execute()

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

col1, col2, col3 = st.columns(3)

with col1:
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

                new_entry = [[name, caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level, weather] for _ in range(10)]
                sheet.append_rows(new_entry)
                
                # ✅ Shuffle Google Sheets Data
                try:
                    data = sheet.get_all_values()  # Get all data
                    headers = data[0]  # Keep headers
                    rows = data[1:]  # Data rows (excluding headers)
                    random.shuffle(rows)  # Shuffle rows

                    # ✅ Clear the sheet & write shuffled data
                    sheet.clear()
                    sheet.append_rows([headers] + rows, value_input_option='RAW')

                    st.success(f"✅ {name} added and Google Sheets shuffled!")
                except Exception as e:
                    st.error(f"⚠️ Error shuffling Google Sheets: {e}")

                st.success(f"✅ {name} added successfully!")
                train_and_update_model()
                st.rerun()

with col2:
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
            try:
                # ✅ Fetch the existing data in Google Sheets
                existing_data = sheet.get_all_records()

                for i, row in enumerate(existing_data, start=2):
                    if row["Coffee Name"] == selected_coffee:  # Find old coffee name
                        updated_row = [new_name, new_caffeine_level, new_sweetness, new_drink_type, new_roast_level,
                                    new_milk_type, new_flavor_notes, new_bitterness_level, new_weather]
                        sheet.update(f'A{i}', [updated_row])  # ✅ Update Google Sheets row

                # ✅ Handle Image Upload
                if image_file:
                    image_path = f"{new_name}.png"
                    with open(image_path, "wb") as f:
                        f.write(image_file.getbuffer())

                    image_link = upload_image_to_drive(image_path, new_name)
                    st.success("📸 Image updated successfully!")

                train_and_update_model()
                st.success(f"✅ {new_name} updated successfully.")
                st.rerun()

            except Exception as e:
                st.error(f"⚠️ Error updating Google Sheets: {e}")

with col3:
    # 🎯 **Delete Coffee**
    st.markdown("### 🗑️ Delete Coffee")

    delete_coffee = st.selectbox("Select coffee to delete:", df["Coffee Name"].dropna().unique())

    if st.button("Delete Coffee"):
        # ✅ Fetch data from Google Sheets
        existing_data = sheet.get_all_records()

        # ✅ Find all rows matching the selected coffee
        rows_to_delete = [i + 2 for i, row in enumerate(existing_data) if row["Coffee Name"] == delete_coffee]

        if rows_to_delete:
            try:
                # ✅ Delete from bottom to top to prevent row shifting issues
                rows_to_delete.sort(reverse=True)
                for row in rows_to_delete:
                    sheet.delete_rows(row)

                st.success(f"🗑 {delete_coffee} deleted successfully from Google Sheets!")

                # ✅ Try to remove the image from Google Drive
                image_link = get_image_url_from_drive(delete_coffee)

                if image_link:
                    # ✅ Correctly extract the file ID
                    image_id = image_link.split("id=")[-1].split("&")[0]  # Only take the ID before "&"

                    # ✅ Delete the file using the correct ID
                    drive_service.files().delete(fileId=image_id).execute()
                    st.success(f"🗑 Image for {delete_coffee} deleted successfully from Google Drive!")

                # ✅ Retrain model after deletion
                train_and_update_model()
                st.rerun()

            except Exception as e:
                st.error(f"⚠️ Error deleting coffee: {e}")

        else:
            st.error("❌ Coffee not found in Google Sheets.")

st.divider()

# ✅ Sidebar Navigation
st.sidebar.markdown("### 🔧 Admin Panel")
if st.sidebar.button("🏠 Back to Menu"):
    st.switch_page("pages/menu.py")
if st.sidebar.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")










