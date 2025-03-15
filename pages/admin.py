import streamlit as st
import requests


st.set_page_config(initial_sidebar_state="collapsed", page_title="Coffee Recommender", layout="wide")  # ✅ Move this to the top

# ✅ Apply custom CSS for a wider main content area

# Flask Backend URL
API_BASE_URL = "https://project-a2bt.onrender.com"

# Function to handle login
def login(username, password):
    response = requests.post(f"{API_BASE_URL}/login", json={"username": username, "password": password})
    if response.status_code == 200:
        return response.json()["access_token"]
    return None

# Function to get admins
def get_admins(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE_URL}/admins", headers=headers)
    return response.json() if response.status_code == 200 else None

# Function to add an admin
def add_admin(token, admin_id, username, password, email):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_BASE_URL}/admins",
        json={"id": admin_id, "username": username, "password": password, "email": email},
        headers=headers,
    )
    return response.json()

# Function to update an admin
def update_admin(token, admin_id, username, email):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.put(
        f"{API_BASE_URL}/admins/{admin_id}",
        json={"username": username, "email": email},
        headers=headers,
    )
    return response.json()

# Function to delete an admin
def delete_admin(token, admin_id):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(f"{API_BASE_URL}/admins/{admin_id}", headers=headers)
    return response.json()

# Streamlit UI
st.title("Admin Login System")

# Initialize session state for token storage
if "token" not in st.session_state:
    st.session_state.token = None

# Login Form
if not st.session_state.token:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            token = login(username, password)
            if token:
                st.session_state.token = token   
                st.success("Login successful!")
                st.switch_page("pages/menu.py")

            else:
                st.error("Invalid username or password")
else:
    st.sidebar.write("✅ Logged in")
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.rerun()


    # Show Admin List
    st.header("Admin Management")
    admins = get_admins(st.session_state.token)

    if admins:
        st.write("### List of Admins")
        for admin in admins:
            st.write(f"🆔 {admin['id']} | 👤 {admin['username']} | 📧 {admin['email']}")

        # Add Admin
        st.write("### Add Admin")
        with st.form("add_admin_form"):
            new_id = st.number_input("ID", min_value=1, step=1)
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            new_email = st.text_input("New Email")
            add_button = st.form_submit_button("Add Admin")

            if add_button:
                result = add_admin(st.session_state.token, new_id, new_username, new_password, new_email)
                if "message" in result:
                    st.success(result["message"])
                else:
                    st.error("Failed to add admin")
                st.rerun()


        # Update Admin
        st.write("### Update Admin")
        with st.form("update_admin_form"):
            update_id = st.number_input("Admin ID to Update", min_value=1, step=1)
            update_username = st.text_input("Updated Username")
            update_email = st.text_input("Updated Email")
            update_button = st.form_submit_button("Update Admin")

            if update_button:
                result = update_admin(st.session_state.token, update_id, update_username, update_email)
                if "message" in result:
                    st.success(result["message"])
                else:
                    st.error("Failed to update admin")
                st.rerun()


        # Delete Admin
        st.write("### Delete Admin")
        delete_id = st.number_input("Admin ID to Delete", min_value=1, step=1)
        if st.button("Delete Admin"):
            result = delete_admin(st.session_state.token, delete_id)
            if "message" in result:
                st.success(result["message"])
            else:
                st.error("Failed to delete admin")
            st.rerun()
        
        st.divider()

            
        if st.button("📜 Go Back to Menu"):
            st.switch_page("pages/menu.py")

        if st.button("🚪 Logout"):
            st.session_state.token = None
            st.switch_page("pages/admin.py")

if st.button("🏠 Go Back to Home"):
            st.switch_page("main.py")
            




