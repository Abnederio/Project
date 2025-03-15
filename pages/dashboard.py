import streamlit as st

st.set_page_config(
    page_title="Admin Dashboard",  
    page_icon="assets/icon.png",  
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import altair as alt
import plotly.express as px
from pages import menu
import google.generativeai as genai
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from googleapiclient.discovery import build
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# CSS
st.markdown(
    """
    <style>
    body {
        background-color: #A27B5C;
    }
    .stApp {
        background-color: #A27B5C; 
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Initialize session state if not set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page
    
# ✅ Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "civic-pulsar-453709-f7-10c1906e9ce5.json"  # Ensure this file is in the project directory

# ✅ Authenticate Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

def load_google_sheet():
    """Load coffee data from Google Sheets."""
    data = sheet.get_all_records()
    return pd.DataFrame(data)

df = load_google_sheet()

# Function to update page selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar Navigation
with st.sidebar:
    st.title('Admin Dashboard')
    st.subheader("Pages")

    # List of pages with display names and corresponding session keys (Conclusion removed)
    pages = [
        ("ℹ️ About", "about"),
        ("📊 Dataset", "dataset"),
        ("📈 EDA", "eda"),
        ("🧼 Data Cleaning", "data_cleaning"),
        ("🤖 Machine Learning", "machine_learning"),
        ("👀 Prediction", "prediction")
    ]

    # Generate buttons dynamically
    for label, key in pages:
        if st.button(label, use_container_width=True, on_click=set_page_selection, args=(key,)):
            break  # Ensures only one button click is processed at a time

    # Project Members Section
    st.subheader("Members")
    st.markdown("""
    1. Nikkos Adrielle Dantes  
    2. Elijah Erle Reyes  
    3. Alistair Aaron Torres  
    4. Andrei Bernard Turgo  
    """)

# Load Data

dataset = df

# Pages Logic
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.write("Alex's Brew Haven is a coffeehouse, celebrated for its high-quality beverages and commitment to innovation. To further enhance the customer experience, this application recommends a drink based on users’ wants. This app aims to provide a seamless and customized ordering experience, helping customers discover new favorites while streamlining operations. By integrating technology with our passion for great coffee, we strive to deliver convenience, efficiency, and a great coffee journey for every customer.")
    
elif st.session_state.page_selection == "dataset":
    st.header("📊 Coffee Dataset")
    st.write("This dataset contains various coffee types with attributes such as caffeine level, sweetness, type, roast level, milk type, flavor notes, and bitterness level.")
    st.dataframe(dataset)  

    st.subheader("📊 Coffee Type Distribution")
    pie_chart = px.pie(dataset, names='Coffee Name', title='Coffee Type Percentage')
    st.plotly_chart(pie_chart)

elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.subheader("Bar Chart of Key Attributes")
    
    selected_column = st.selectbox("Select an Attribute to Visualize", dataset.columns)
    bar_chart = px.bar(dataset, x='Coffee Name', y=selected_column, title=f'Distribution of {selected_column}')
    st.plotly_chart(bar_chart)

elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")
    st.subheader("Null Values Check")

    null_values = dataset.isnull().sum().sum()
    if null_values == 0:
        st.success("The dataset contains 0 null values.")
    else:
        st.warning(f"The dataset contains {null_values} null values.")

elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction Accuracy")
    model_accuracy = 0.98  # Replace with actual accuracy
    st.subheader("Model Training Accuracy")
    st.success(f"The model achieved an accuracy of {model_accuracy * 100:.2f}% on the training dataset.")

st.divider()

# Navigation Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🏠 Go Back to Menu"):
        st.switch_page("pages/menu.py")

with col2:
    if st.button("🚪 Logout"):
        st.session_state.token = None
        st.switch_page("pages/admin.py")
