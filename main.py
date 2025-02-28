import pandas as pd
import catboost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib
import os
import numpy as np



df = pd.read_csv("coffee_dataset.csv")

df.head()

X = df.drop(columns=['Coffee Name'])
y = df['Coffee Name']

X.fillna("Unknown", inplace=True)
y.fillna("Unknown", inplace=True)

cat_features = list(range(X.shape[1]))  # Ensure correct categorical indices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MODEL_PATH = "catboost_model.pkl"
ACCURACY_PATH = "catboost_accuracy.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(ACCURACY_PATH):
    model = joblib.load(MODEL_PATH)  # Load pre-trained model
    accuracy = joblib.load(ACCURACY_PATH)
else:
    model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=9, verbose=0)
    model.fit(X_train, y_train, cat_features=cat_features)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(model, MODEL_PATH)  # Save the trained model
    joblib.dump(accuracy, ACCURACY_PATH)

st.write(f"**Model Accuracy:** {accuracy:.2f}%")

st.header(" Prediction")

# Input boxes for the features

caffeine_level = st.selectbox(
    'How would you like the Caffeine Level?',
    ('Low', 'Medium', 'High')
)

sweetness = st.selectbox(
    'How would you like the Sweetness?',
    ('Low', 'Medium', 'High')
)

drink_type = st.selectbox(
    'How would you like the drink type?',
    ('Frozen', 'Iced', 'Hot')
)

roast_level = st.selectbox(
    'How would you like the roast level?',
    ('Medium', 'None', 'Dark')
)

milk_type = st.toggle("Do you want milk in your coffee?")
if milk_type: 
    milk_type = 'Dairy'
else:
    milk_type = 'None'
    
flavor_notes = st.selectbox(
    'How would you like the Flavor?',
    ('Creamy', 'Earthy', 'Vanilla', 'Sweet', 'Bold', 'Bitter', 'Smooth', 'Nutty'
 'Chocolate', 'Caramel', 'Espresso')
)

bitterness_level = st.selectbox(
    'How would you like the bitterness Level?',
    ('Low', 'Medium', 'High')
)

classes_list = [
    'Handcrafted Milk Chocolate', 'Mocha Latte', 'French Vanilla',
    'Freshly Brewed Iced Tea', 'Irish Coffee', 'Flat White', 'Café Mocha',
    'Dead Eye', 'Iced Cappuccino', 'Spanish Latte', 'Caramel Macchiato',
    'Triple Coffee Jelly', 'Cafe Americano', 'French Vanilla Cold Brew',
    'Cortado', 'Vienna Coffee', 'Latte', 'Turkish Coffee', 'Affogato',
    'Iced Cappuccino Supreme', 'Iced Coffee', 'Iced French Vanilla', 'Ristretto',
    'Dark Roast', 'Classic Roast', 'Espresso', 'Double Double', 'Doppio',
    'Iced Latte', 'Macchiato', 'Cold Brew', 'Nitro Cold Brew',
    'Frozen Hot Chocolate', 'Cappuccino', 'Black Eye', 'Red Eye',
    'Iced Handcrafted Milk Chocolate'
]
        
        # Button to detect
if st.button('Recommend', key='rfr_detect'):
    # Prepare the input data for prediction
    rfr_input_data = [[caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level]]
    
    # Predict the recommended coffee
    rfr_prediction = model.predict(rfr_input_data)

    # Extract the prediction properly
    recommended_coffee = rfr_prediction[0]  # First element

    # Ensure it's a plain string (handles NumPy array cases)
    if isinstance(recommended_coffee, (list, np.ndarray)):  
        recommended_coffee = recommended_coffee[0]  # Extract string if it's in a list/array

    recommended_coffee = str(recommended_coffee)  # Convert to string
    
    # Display the recommendation
    st.markdown(f'The coffee we recommend is: `{recommended_coffee}`')

    # Construct the image path
    image_path = f"images/{recommended_coffee}.png"

    # Display the image
    st.image(image_path, caption=f"Recommended Coffee: {recommended_coffee}", use_column_width=True)
    


