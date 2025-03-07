import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import menu


# Page configuration
st.set_page_config(
    page_title="Admin Dashboard", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Admin Dashboard')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Nikkos Adrielle Dantes\n2. Elijah Erle Reyes\n3. Alistair Aaron Torres\n4. Andrei Bernard Turgo")

#######################
# Data

# Load data
dataset = pd.read_csv("coffee_dataset.csv")
#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Coffee Dataset")
    
    st.write("This dataset contains various coffee types with attributes such as caffeine level, sweetness, type, roast level, milk type, flavor notes, and bitterness level.")
    
    st.dataframe(dataset)  # Display the dataset as a table
    
    # Pie Chart
    st.subheader("📊 Coffee Type Distribution")
    pie_chart = px.pie(dataset, names='Coffee Name', title='Coffee Type Percentage')
    st.plotly_chart(pie_chart)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")
    
    st.subheader("Bar Chart of Key Attributes")
    
    selected_column = st.selectbox("Select an Attribute to Visualize", dataset.columns)
    
    bar_chart = px.bar(dataset, x='Coffee Name', y=selected_column, title=f'Distribution of {selected_column}')
    st.plotly_chart(bar_chart)

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
    
st.divider()

if st.button("🏠 Go Back to Menu"):
    st.switch_page("pages/menu.py")

if st.button("🚪 Logout"):
    st.session_state.token = None
    st.switch_page("pages/admin.py")
