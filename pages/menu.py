import streamlit as st

# ✅ Move config to the top
st.set_page_config(page_title="Admin Menu", layout="wide")

# ✅ Custom CSS for a cleaner UI
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }  /* Hide Sidebar */
        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            background-color: #0078D4;
            color: white;
            font-size: 18px;
            padding: 10px;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Ensure user is logged in
if "token" not in st.session_state or not st.session_state.token:
    st.switch_page("pages/admin.py")

# ✅ Title with spacing
st.title("☕ Admin Dashboard")
st.markdown("### Manage everything from here")

# ✅ Centered menu buttons
st.divider()  # Add a divider for better separation

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

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
    if st.button("🚪 Logout"):
        st.session_state.token = None  # Clear token
        st.switch_page("pages/admin.py")

st.divider() 
        
        