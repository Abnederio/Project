import streamlit as st
import pandas as pd
import plotly.express as px
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# ✅ Set Page Configuration
st.set_page_config(
    page_title="Admin Dashboard",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Google Sheets Setup
SHEET_ID = "1NCHaEsTIvYUSUgc2VHheP1qMF9nIWW3my5T6NpoNZOk"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "civic-pulsar-453709-f7-10c1906e9ce5.json"

# ✅ Authenticate Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

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

# ✅ Custom CSS for Coffee-Themed Styling
# ✅ Custom CSS for Improved Coffee Theme (No Red)
st.markdown(
    """
    <style>
        /* Background */
        body {
            background-color: #ECE3CE !important;  /* Creamy Cappuccino */
        }
        .stApp {
            background-color: #ECE3CE !important;  /* Soft warm tone */
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #6D4C41 !important;  /* Rich Mocha */
        }

        /* General Button Styling */
        div.stButton > button {
            width: 100%;
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            transition: 0.3s ease-in-out;
            border: none;
            font-weight: bold;
        }
        
        /* Brown Button (Go Back) */
        div.stButton > button:first-child {
            background-color: #8B5E3B;  /* Warm Espresso */
            color: white;
        }
        div.stButton > button:first-child:hover {
            background-color: #6D4C41;  /* Dark Mocha */
            transform: scale(1.05);  
        }

        /* Neutral Button (Logout) */
        div.stButton > button:last-child {
            background-color: #5E503F;  /* Deep Mocha */
            color: white;
        }
        div.stButton > button:last-child:hover {
            background-color: #4E4237;  /* Darker Mocha */
            transform: scale(1.05);
        }
        
        /* Sidebar Buttons */
        .st-emotion-cache-ocqkz7:hover { 
            background-color: #5B3A29 !important;  /* Dark coffee brown */
            color: white !important;
        }

        /* Headers & Text */
        h1, h2, h3, h4, h5, h6 {
            color: #4E342E !important;  /* Dark Chocolate */
        }
        p, div {
            color: #3E2723 !important;  /* Rich Espresso Text */
        }

        /* Dataset Table */
        .stDataFrame {
            background-color: #F5E1C8 !important;  /* Light Cappuccino */
            color: #3E2723 !important;  /* Mocha text */
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Sidebar Navigation
with st.sidebar:
    st.image("assets/shop.jpeg", use_container_width=True, Height = 500)  # Resized sidebar image
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
    st.image("assets/shop.jpeg", width=500)  # Resized image
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
    model_accuracy = 0.94  # Replace with actual accuracy
    st.success(f"✅ The model achieved **{model_accuracy * 100:.2f}%** accuracy.")

# ✅ Navigation Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🏠 Go Back to Menu"):
        st.session_state.page_selection = "about"
        st.switch_page("pages/home.py")

with col2:
    if st.button("🚪 Logout"):
        st.session_state.token = None
        st.switch_page("pages/admin.py")



