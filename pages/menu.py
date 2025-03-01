import streamlit as st

st.set_page_config(page_title="Admin Menu", layout="wide")

# Ensure user is logged in
if "token" not in st.session_state or not st.session_state.token:
    st.warning("Please login first.")
    st.switch_page("pages/admin.py")

st.title("☕ Admin Dashboard")

# Menu buttons
col1, col2, col3, col4 = st.columns(4)

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