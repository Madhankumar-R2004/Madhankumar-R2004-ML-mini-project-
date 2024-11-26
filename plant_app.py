# Streamlit App Code
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('crop_yield_prediction_model.pkl')

# Prediction function with dynamic one-hot encoding
def predict_yield(crop_name, crop_year, season_name, state_name, area, production, rainfall, fertilizer, pesticide):
    # Create a dictionary of the inputs
    input_data = {
        'Crop': crop_name,
        'Crop_Year': crop_year,
        'Season': season_name,
        'State': state_name,
        'Area': area,
        'Production': production,
        'Annual_Rainfall': rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide
    }
    
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Perform one-hot encoding on input_df to match the model's training structure
    input_encoded = pd.get_dummies(input_df, columns=['Crop', 'Season', 'State'])

    # Align the input DataFrame with the training features
    missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0  # Add missing columns as zeros
    input_encoded = input_encoded[model.feature_names_in_]

    # Predict crop yield
    prediction = model.predict(input_encoded)
    return prediction[0]

# Streamlit UI
st.title("Crop Yield Prediction")

# Input fields
crop_name = st.text_input("Crop Name")
crop_year = st.number_input("Crop Year", min_value=1990, max_value=2023)
season_name = st.text_input("Season")
state_name = st.text_input("State")
area = st.number_input("Area", min_value=0.0, format="%.2f")
production = st.number_input("Production", min_value=0.0, format="%.2f")
rainfall = st.number_input("Annual Rainfall", min_value=0.0, format="%.2f")
fertilizer = st.number_input("Fertilizer", min_value=0.0, format="%.2f")
pesticide = st.number_input("Pesticide", min_value=0.0, format="%.2f")

# Make prediction
if st.button("Predict Yield"):
    prediction = predict_yield(crop_name, crop_year, season_name, state_name, area, production, rainfall, fertilizer, pesticide)
    st.success(f"Predicted Crop Yield: {prediction:.2f}")
