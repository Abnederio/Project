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
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("coffee_dataset.csv")

# Preview dataset
st.write("## Coffee Dataset Preview")
st.write(df.head())

# Separate features (X) and target variable (y)
X = df.drop(columns=['Coffee Name'])
y = df['Coffee Name']

# Handle missing values
X.fillna("Unknown", inplace=True)
y.fillna("Unknown", inplace=True)

# ✅ Fix: Convert all target labels (y) to string before encoding
y = y.astype(str)

# Encode target labels (fixes CatBoostError)
le = LabelEncoder()
y = le.fit_transform(y)  # Convert labels to numeric

# Save label encoder for decoding predictions later
LABEL_ENCODER_PATH = "label_encoder.pkl"
joblib.dump(le, LABEL_ENCODER_PATH)

# Identify categorical features for CatBoost
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define paths for saving/loading model and accuracy
MODEL_PATH = "catboost_model.pkl"
ACCURACY_PATH = "catboost_accuracy.pkl"

# Check if trained model exists, else train a new one
if os.path.exists(MODEL_PATH) and os.path.exists(ACCURACY_PATH):
    model = joblib.load(MODEL_PATH)  # Load trained model
    accuracy = joblib.load(ACCURACY_PATH)
else:
    model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=9, verbose=0)
    model.fit(X_train, y_train, cat_features=cat_features)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save the trained model and accuracy
    joblib.dump(model, MODEL_PATH)
    joblib.dump(accuracy, ACCURACY_PATH)

st.write(f"**Model Accuracy:** {accuracy:.2f}")

# --- Feature Importance Section ---
if st.button("Show Feature Importance"):
    feature_importances = model.get_feature_importance(Pool(X_train, label=y_train, cat_features=cat_features))
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    st.write("### Feature Importance")
    st.dataframe(importance_df)  # Display as a table

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='royalblue')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance in CatBoost Model')
    ax.invert_yaxis()
    st.pyplot(fig)  # Display plot in Streamlit

# --- Coffee Recommendation System ---
st.header("Coffee Recommendation System")

# Input options for features
caffeine_level = st.selectbox('Caffeine Level', ('Low', 'Medium', 'High'))
sweetness = st.selectbox('Sweetness', ('Low', 'Medium', 'High'))
drink_type = st.selectbox('Drink Type', ('Frozen', 'Iced', 'Hot'))
roast_level = st.selectbox('Roast Level', ('Medium', 'None', 'Dark'))

milk_type = 'Dairy' if st.toggle("Do you want milk?") else 'None'

flavor_notes = st.selectbox(
    'Flavor Notes',
    ('Creamy', 'Earthy', 'Vanilla', 'Sweet', 'Bold', 'Bitter', 'Smooth', 'Nutty',
     'Chocolate', 'Caramel', 'Espresso')
)

bitterness_level = st.selectbox('Bitterness Level', ('Low', 'Medium', 'High'))

# Button to make prediction
if st.button('Recommend Coffee'):
    # Prepare input data for prediction
    input_data = [[caffeine_level, sweetness, drink_type, roast_level, milk_type, flavor_notes, bitterness_level]]
    
    # Predict using the model
    prediction = model.predict(input_data)
    
    # Decode the prediction back to the coffee name
    le = joblib.load(LABEL_ENCODER_PATH)  # Load label encoder
    recommended_coffee = le.inverse_transform([prediction[0]])[0]

    # Display recommendation
    st.success(f"☕ The coffee we recommend is: **{recommended_coffee}**")

    # Construct image path
    image_path = f"images/{recommended_coffee}.png"

    # Check if image exists before displaying
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended Coffee: {recommended_coffee}", use_column_width=True)
    else:
        st.warning("Image not available for this coffee.")

