import streamlit as st

st.title("🔧 Admin Panel")
st.write("Welcome to the admin dashboard!")

if st.button("Go Back"):
    st.query_params.clear()  # Clears all query parameters
    st.rerun()  # Refreshes the app to return to `main.py`

