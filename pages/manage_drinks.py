import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")

# 🔹 Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"  # Your Google Sheet ID
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
CREDS_FILE = "civic-pulsar-453709-f7-10c1906e9ce5.json"  # Your Google API credentials

# 🔹 Authenticate Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1  # Access the first sheet

# 🔹 Google Drive Setup
FOLDER_ID = "1GtQVlpBSe71mvDk5fbkICqMdUuyfyGGn"  # Your Google Drive Folder ID

def authenticate_drive():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    gauth = GoogleAuth()
    gauth.credentials = creds
    return GoogleDrive(gauth)

def upload_image_to_drive(image_path, image_name):
    drive = authenticate_drive()

    # ✅ Upload file to the specific folder
    file_drive = drive.CreateFile({'title': image_name, 'parents': [{'id': FOLDER_ID}]})
    file_drive.SetContentFile(image_path)
    file_drive.Upload()

    # ✅ Make the file public
    file_drive.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})

    # ✅ Get the direct image link
    image_id = file_drive['id']
    image_link = f"https://drive.google.com/uc?id={image_id}"  # Direct image link

    return image_link

# 🔹 Load Data from Google Sheets
def load_google_sheet():
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# 🔹 Save Data to Google Sheets
def save_to_google_sheet(df):
    try:
        # Ensure all data is string and remove any unwanted columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')].astype(str).fillna("")

        # Convert the DataFrame to a list of lists (to match the structure expected by Google Sheets API)
        new_entry_list = df.values.tolist()  # Convert to list of lists

        # Append new rows to Google Sheets, using proper headers
        sheet.append_rows(new_entry_list, value_input_option='RAW')
        st.success("Google Sheets updated successfully!")

        return True  # Return True if successful

    except Exception as e:
        st.error(f"Error updating Google Sheets: {e}")
        return False  # Return False if there was an error

# 🔹 Train & Update Model
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

df = load_google_sheet()

# 🔹 Convert Image URLs into Clickable Images
def image_formatter(url):
    return f'<img src="{url}" width="100">'

# 🔹 Show Dataframe with Images
st.markdown("### ☕ Current Coffee Menu")
st.dataframe(df, height=500)

st.divider()

# 🎨 **Columns for Better Layout**
col1, col2, col3 = st.columns([2, 2, 1])

# ➕ **Add Coffee** in col1
# ➕ **Add Coffee** in col1
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

        image_file = st.file_uploader("Upload an image for the coffee", type=['jpg', 'jpeg', 'png'])

        submit = st.form_submit_button("Add Coffee")

        if submit:
            if not name:
                st.error("❌ Coffee Name is required!")
            elif name in df["Coffee Name"].values:
                st.error("⚠️ Coffee already exists!")
            else:
                image_path = f"{name}.png"  # Use the coffee name for the image path
                image_link = f"{name}.png"

                if image_file:
                    with open(image_path, "wb") as f:
                        f.write(image_file.getbuffer())
                    image_link = upload_image_to_drive(image_path, image_path)
                    st.success("📸 Image uploaded successfully!")

                # Create the new coffee entry as a DataFrame
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

                # Convert the new coffee entry to a list of lists (to match the structure expected by Google Sheets API)
                new_entry_list = new_entry.values.tolist()

                # Append only the new rows to Google Sheets
                sheet.append_rows(new_entry_list, value_input_option='RAW')

                train_and_update_model()
                st.success(f"☕ {name} added successfully!")
                st.rerun()

# ✏️ **Update Coffee** in col2
# ✏️ **Update Coffee** in col2
with col2:
    st.markdown("### ✏️ Update Coffee")
    coffee_names = df["Coffee Name"].dropna().unique()
    selected_coffee = st.selectbox("Select coffee to update:", coffee_names)

    if selected_coffee:
        coffee_data = df[df["Coffee Name"] == selected_coffee].iloc[0]

        # Get the current values for all features
        new_name = st.text_input("Coffee Name", value=coffee_data["Coffee Name"])
        new_caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Caffeine Level"]))
        new_sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Sweetness"]))
        new_drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'], index=['Frozen', 'Iced', 'Hot'].index(coffee_data["Type"]))
        new_roast_level = st.selectbox('Roast Level:', ['Medium', 'None', 'Dark'], index=['Medium', 'None', 'Dark'].index(coffee_data["Roast Level"]))
        new_milk_type = 'Dairy' if coffee_data["Milk Type"] == "Dairy" else 'No Dairy'
        new_flavor_notes = st.selectbox('Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'], index=['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'].index(coffee_data["Flavor Notes"]))
        new_bitterness_level = st.selectbox('Bitterness Level:', ['Low', 'Medium', 'High'], index=['Low', 'Medium', 'High'].index(coffee_data["Bitterness Level"]))
        new_weather = st.selectbox('Weather:', ['Hot', 'Cold'], index=['Hot', 'Cold'].index(coffee_data["Weather"]))

        # Upload new image if provided
        image_file = st.file_uploader("Upload a new image for the coffee", type=['jpg', 'jpeg', 'png'])

        if st.button("Update Coffee"):
            # Update the coffee in the DataFrame
            df.loc[df["Coffee Name"] == selected_coffee, ["Coffee Name", "Caffeine Level", "Sweetness", "Type", "Roast Level", "Milk Type", "Flavor Notes", "Bitterness Level", "Weather"]] = [new_name, new_caffeine_level, new_sweetness, new_drink_type, new_roast_level, new_milk_type, new_flavor_notes, new_bitterness_level, new_weather]

            # Handle image update
            if image_file:
                image_path = f"{new_name}.png"  # Use the name directly, without replacing spaces with underscores
                with open(image_path, "wb") as f:
                    f.write(image_file.getbuffer())
                image_link = upload_image_to_drive(image_path, image_path)
                
                # Update the image link in the DataFrame
                df.loc[df["Coffee Name"] == selected_coffee, "Image"] = image_link
                st.success("📸 Image updated successfully!")

            # Fetch the existing data in Google Sheets
            existing_data = sheet.get_all_records()

            # Iterate over all rows and update rows where the coffee name matches
            for i, row in enumerate(existing_data, start=2):  # starting at 2 because Sheet rows start from 1
                if row["Coffee Name"] == selected_coffee:
                    # Prepare the updated row data to match the columns
                    updated_row = df[df["Coffee Name"] == selected_coffee].iloc[0].values.tolist()

                    # Update the row in Google Sheets
                    sheet.update(f'A{i}', [updated_row])  # Use f-string to reference the specific row

            st.success(f"✅ {new_name} updated successfully in Google Sheets!")
            st.rerun()

# 🗑 **Delete Coffee** in col3
with col3:
    st.markdown("### 🗑 Delete Coffee")
    delete_coffee = st.selectbox("Select coffee to delete:", df["Coffee Name"].dropna().unique())

    if st.button("Delete Coffee"):
        df = df[df["Coffee Name"] != delete_coffee]
        save_to_google_sheet(df)

        train_and_update_model()
        st.success(f"🗑 {delete_coffee} deleted successfully!")
        st.rerun()

st.divider()

if st.button("🏠 Go Back to Menu"):
    st.switch_page("pages/menu.py")

if st.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")






