import pandas as pd
import catboost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib
import os


df = pd.read_csv("Balanced_CoffeeDS_500.csv")

df = df.drop(columns=['Acidity Level']) #might be dropped
df = df.drop(columns=['Temperature'])

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

df["Caffeine Level"].unique()

caffeine_level = st.selectbox(
    'How would you like your Caffeine Level?',
    ('Low', 'Medium', 'High')
)

sweetness = st.selectbox(
    'How would you like your Sweetness?',
    ('Low', 'Medium', 'High')
)

drink_type = st.selectbox(
    'How would you like the Type?',
    ('Frozen', 'Hot', 'Iced')
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
    ('Vanilla', 'Coffee', 'Chocolate', 'Nutty', 'Sweet', 'Bitter', 'Creamy', 'Earthy',
     'Caramel', 'Espresso')
)

bitterness_level = st.selectbox(
    'How would you like your bitterness Level?',
    ('Low', 'Medium', 'High')
)

weather = st.selectbox(
    'How do you feel the ambiance today?',
    ('Hot', 'Cold')
)

classes_list = [
    'Cappuccino', 'Double Double', 'Dark Roast', 'Cafe Americano', 
    'French Vanilla', 'Macchiato', 'Latte', 'Handcrafted milk chocolate', 
    'Cafe Mocha', 'Classic Roast', 'Iced Capp', 'Freshly Brewed Iced Tea', 
    'Frozen Hot Chocolate', 'Triple Coffee Jelly', 'Iced Capp Supreme', 
    'Iced Coffee', 'Iced French Vanilla', 'French Vanilla Cold Brew', 
    'Spanish Latte', 'Iced Citrus Black Tea', 'Iced Latte', 
    'Iced Handcrafted Milk Chocolate'
]
        
        # Button to detect the Iris species
if st.button('Recommend', key='rfr_detect'):
            # Prepare the input data for prediction
    rfr_input_data = [[caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level, weather]]
            
            # Predict the Iris species
    rfr_prediction = model.predict(rfr_input_data)
            
            # Display the prediction result
    st.markdown(f'The coffee we recommend is: `{rfr_prediction[0]}`')
