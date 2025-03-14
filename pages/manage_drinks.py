import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
import gspread

st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")

# 🔹 Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"  # Your Google Sheet ID
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "credentials.json"  # Your Google API credentials

# 🔹 Authenticate with Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1  # Access the first sheet

# 🔹 Load Data from Google Sheets
def load_google_sheet():
    data = sheet.get_all_records()
    return pd.DataFrame(data)

df = load_google_sheet()

# 🔹 Save Data to Google Sheets
def save_to_google_sheet(df):
    sheet.clear()  # Clear existing data
    sheet.update([df.columns.values.tolist()] + df.values.tolist())  # Upload new data

st.title("🥤 Manage Drinks")
st.markdown("### ☕ Current Coffee Menu")
st.dataframe(df)  # Display Google Sheets data in Streamlit

st.divider()  # Separate sections

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

                df = pd.concat([new_entry, df], ignore_index=True)
                df = df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset
                save_to_google_sheet(df)  # ✅ Save directly to Google Sheets

                st.success(f"☕ {name} added successfully!")
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

            save_to_google_sheet(df)  # ✅ Save directly to Google Sheets
            st.success(f"✅ {selected_coffee} updated successfully!")
            st.rerun()

# 🗑 **Delete Coffee**
with col3:
    st.markdown("### 🗑 Delete Coffee")
    delete_coffee = st.selectbox("Select coffee to delete:", df["Coffee Name"].dropna().unique())

    if st.button("Delete Coffee"):
        df = df[df["Coffee Name"] != delete_coffee]
        save_to_google_sheet(df)  # ✅ Save directly to Google Sheets

        st.success(f"🗑 {delete_coffee} deleted successfully!")
        st.rerun()

st.divider()

if st.button("🏠 Go Back to Menu"):
    st.switch_page("pages/menu.py")

if st.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")


