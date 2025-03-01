import streamlit as st
import pandas as pd

# Load existing coffee dataset
df = pd.read_csv("coffee_dataset.csv")

st.title("🥤 Manage Drinks")

# Show list of drinks
st.write("### Current Coffee Menu")
st.dataframe(df)

# Add new coffee
with st.form("add_coffee"):
    st.write("### Add New Coffee")
    name = st.text_input("Coffee Name")
    caffeine_level = st.selectbox('Caffeine Level:', ['Low', 'Medium', 'High'])
    sweetness = st.selectbox('Sweetness:', ['Low', 'Medium', 'High'])
    drink_type = st.selectbox('Drink Type:', ['Frozen', 'Iced', 'Hot'])
    roast_level = st.selectbox('Roast Level:', ['Medium', 'None', 'Dark'])
    milk_type = 'Dairy' if st.toggle("Do you want milk?") else 'None'
    flavor_notes = st.selectbox('Flavor Notes:', ['Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy', 'Caramel', 'Espresso'])
    bitterness_level = st.selectbox('Bitterness Level:', ['Low', 'Medium', 'High'])
    weather = st.selectbox('Weather:', ['Hot', 'Cold'])

    submit = st.form_submit_button("Add Coffee")  # ✅ Added missing submit button

    if submit:
        # ✅ Ensure new entry follows dataset structure
        new_entry = pd.DataFrame([{
            "Coffee Name": name,
            "Caffeine Level": caffeine_level,
            "Sweetness": sweetness,
            "Drink Type": drink_type,
            "Roast Level": roast_level,
            "Milk Type": milk_type,
            "Flavor Notes": flavor_notes,
            "Bitterness Level": bitterness_level,
            "Weather": weather
        }])

        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv("coffee_dataset.csv", index=False)

        st.success(f"☕ {name} added successfully!")
        st.rerun()