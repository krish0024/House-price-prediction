
# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction")

# Use the current working directory in Colab
MODEL_PATH = "house_price_pipeline.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# Try to recover input feature names from model if possible
def get_feature_list(model):
    # If user trained with a DataFrame pipeline, we can try to inspect transformers
    # Fallback: ask the user to enter features manually below
    try:
        # If preprocessor used OneHotEncoder with feature_names_in_ or similar
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            # Best-effort: read original feature names via attributes
            if hasattr(model.named_steps['preprocessor'], 'feature_names_in_'):
                return list(model.named_steps['preprocessor'].feature_names_in_)
    except Exception:
        pass
    return None

feature_list = get_feature_list(model)

st.write("Enter the input values for the house features below.")

# If feature list discovered, create inputs automatically; else use manual defaults.
if feature_list:
    inputs = {}
    for f in feature_list:
        # simple heuristics for input type
        if 'bed' in f.lower() or 'room' in f.lower():
            inputs[f] = st.number_input(f, min_value=0.0, value=3.0, step=1.0)
        elif 'area' in f.lower() or 'sqft' in f.lower():
             inputs[f] = st.number_input(f, min_value=0.0, value=1000.0)
        elif X[f].dtype in ['int64', 'float64']:
             inputs[f] = st.number_input(f, min_value=0.0, value=X[f].mean())
        elif X[f].dtype in ['object', 'category']:
             inputs[f] = st.selectbox(f, options=X[f].unique().tolist())
        else:
            inputs[f] = st.text_input(f)

    X_pred = pd.DataFrame([inputs])
else:
    # Default manual fields — edit these to match your dataset
    # Based on the columns in your dataframe, I'll create some default inputs.
    st.write("Could not automatically determine features. Please enter values for the following:")
    area = st.number_input("area", min_value=0.0, value=df['area'].mean())
    bedrooms = st.number_input("bedrooms", min_value=0, value=int(df['bedrooms'].mean()))
    bathrooms = st.number_input("bathrooms", min_value=0.0, value=df['bathrooms'].mean())
    stories = st.number_input("stories", min_value=0, value=int(df['stories'].mean()))
    mainroad = st.selectbox("mainroad", options=df['mainroad'].unique().tolist())
    guestroom = st.selectbox("guestroom", options=df['guestroom'].unique().tolist())
    basement = st.selectbox("basement", options=df['basement'].unique().tolist())
    hotwaterheating = st.selectbox("hotwaterheating", options=df['hotwaterheating'].unique().tolist())
    airconditioning = st.selectbox("airconditioning", options=df['airconditioning'].unique().tolist())
    parking = st.number_input("parking", min_value=0, value=int(df['parking'].mean()))
    prefarea = st.selectbox("prefarea", options=df['prefarea'].unique().tolist())
    furnishingstatus = st.selectbox("furnishingstatus", options=df['furnishingstatus'].unique().tolist())

    X_pred = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]],
                          columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'])


if st.button("Predict"):
    try:
        pred = model.predict(X_pred)[0]
        st.success(f"Predicted price: {pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Check that the UI fields match the features the model expects.")
