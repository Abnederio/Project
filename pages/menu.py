import streamlit as st
import requests

# ✅ Move config to the top
st.set_page_config(page_title="Admin Menu", layout="wide")

API_URL = "https://project-a2bt.onrender.com"

# ✅ Custom CSS for a cleaner UI
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

# ✅ Ensure user is logged in
if "token" not in st.session_state or not st.session_state.token:
    st.switch_page("pages/admin.py")

# ✅ Title with spacing
st.title("☕ Admin Dashboard")
st.markdown("### Manage and monitor: ")

# ✅ Centered menu buttons
st.divider()  # Add a divider for better separation

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col1:
    if st.button("📊 Dashboard"):
        st.switch_page("pages/dashboard.py")

with col2:
    if st.button("🥤 Manage Drinks"):
        st.switch_page("pages/manage_drinks.py")

with col3:
    if st.button("🛠️ Manage Admins"):
        st.switch_page("pages/admin.py")

with col4:
    if st.button("📡 Check API status"):
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            st.success("✅ API is working!")
        else:
            st.error("❌ API failed to respond.")

with col5:
    if st.button("🚪 Logout"):
        st.session_state.token = None  # Clear token
        st.switch_page("pages/admin.py")

st.divider() 
        
        