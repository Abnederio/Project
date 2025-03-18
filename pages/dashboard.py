import streamlit as st
import pandas as pd
import plotly.express as px
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import joblib
from sklearn.metrics import accuracy_score
from googleapiclient.discovery import build

# ✅ Set Page Configuration
st.set_page_config(
    page_title="Admin Dashboard",  
    layout="wide",
    initial_sidebar_state="expanded"
)

if "token" not in st.session_state or not st.session_state.token:
    st.switch_page("pages/admin.py")

# ✅ Load Google API Credentials Securely (from Streamlit Secrets)
if "GOOGLE_CREDENTIALS" not in st.secrets:
    st.error("❌ GOOGLE_CREDENTIALS not found! Set up secrets in Streamlit Cloud.")
    st.stop()

google_creds = st.secrets["GOOGLE_CREDENTIALS"] 
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    google_creds, ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
)

# ✅ Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# ✅ Google Drive Setup (For Image Retrieval)
FOLDER_ID = "1GtQVlpBSe71mvDk5fbkICqMdUuyfyGGn"
drive_service = build("drive", "v3", credentials=creds)

def load_google_sheet():
    """Load coffee data from Google Sheets."""
    data = sheet.get_all_records()
    return pd.DataFrame(data)

df = load_google_sheet()

# ✅ Initialize session state if not set
if "page_selection" not in st.session_state:
    st.session_state.page_selection = "about"  # Default page

# ✅ Function to update page selection
def set_page_selection(page):
    st.session_state.page_selection = page

# ✅ Custom CSS for Perfect Coffee Shop Theme
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

        /* Dataset Table */
        .stDataFrame {
            background-color: #F0D9B5 !important;  /* Light Cappuccino */
            color: #3E2723 !important;  /* Mocha text */
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Sidebar Navigation
with st.sidebar:  
    st.title("☕ Admin Dashboard")
    st.subheader("📌 Pages")

    # Define pages
    pages = {
        "ℹ️ About": "about",
        "📊 Dataset": "dataset",
        "📈 EDA": "eda",
        "🧼 Data Cleaning": "data_cleaning",
        "🤖 Machine Learning": "machine_learning",
        "👀 Prediction": "prediction"
    }

    # ✅ Sidebar Buttons
    for label, key in pages.items():
        if st.button(label, key=key, use_container_width=True):
            st.session_state.page_selection = key

    st.markdown("---")  # Divider

    # ✅ Project Members Section
    st.subheader("👥 Members")
    st.markdown("""
    - **Nikkos Adrielle Dantes**  
    - **Elijah Erle Reyes**  
    - **Alistair Aaron Torres**  
    - **Andrei Bernard Turgo**  
    """)

# ✅ Load Data
dataset = df

# ✅ Page Logic (Switch Pages)
if st.session_state.page_selection == "about":
    st.image("assets/shop.jpeg", use_container_width=True)  # Resized image
    st.header("ℹ️ Welcome to Alex's Brew Haven ☕")
    st.write("""
    **Alex's Brew Haven** is a coffeehouse known for its **premium coffee** and **innovative flavors**.  
    This application enhances the customer experience by **recommending personalized drinks** based on preferences.  
    """)
    
    st.markdown("""
    ### **✨ Why Choose Us?**
    - **🌱 Organic & Sustainable Coffee**
    - **👨‍🔬 Expertly Crafted Recipes**
    - **📲 Seamless & Smart Ordering**
    - **🎯 AI-Powered Drink Recommendations**
    """)

elif st.session_state.page_selection == "dataset":
    st.header("📊 Coffee Dataset")
    st.write("This dataset contains various coffee types with attributes such as caffeine level, sweetness, type, roast level, milk type, flavor notes, and bitterness level.")
    st.dataframe(dataset)  
    st.subheader("📊 Coffee Type Distribution")
    pie_chart = px.pie(dataset, names="Coffee Name", title="Coffee Type Percentage")
    st.plotly_chart(pie_chart, use_container_width=True)

elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")
    selected_column = st.selectbox("Choose an Attribute:", dataset.columns)
    bar_chart = px.bar(dataset, x="Coffee Name", y=selected_column, title=f"Distribution of {selected_column}")
    st.plotly_chart(bar_chart, use_container_width=True)

elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning & Pre-processing")
    null_values = dataset.isnull().sum().sum()
    if null_values == 0:
        st.success("✅ The dataset contains **0** null values.")
    else:
        st.warning(f"⚠️ The dataset contains **{null_values}** null values.")

elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")
    st.write("🚀 **Machine Learning Model Implementation Coming Soon...**")

elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction Accuracy")
    accuracy = joblib.load("catboost_accuracy.pkl")
    st.success(f"✅ The model achieved **{accuracy * 100:.2f}%** accuracy.")


# Sidebar Navigation
st.sidebar.markdown("### 🔧 Admin Panel")
if st.sidebar.button("🏠 Back to Menu"):
    st.switch_page("pages/menu.py")
if st.sidebar.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")



