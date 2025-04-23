import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'best_model.pkl' is in the same directory.")
    st.stop()

st.title("🧠 Autism Prediction App")
st.write("Answer the following screening questions to predict the likelihood of Autism Spectrum Disorder (ASD).")

# Screening Questions and Inputs
A1 = st.selectbox("A1 Score", [0, 1])
A2 = st.selectbox("A2 Score", [0, 1])
A3 = st.selectbox("A3 Score", [0, 1])
A4 = st.selectbox("A4 Score", [0, 1])
A5 = st.selectbox("A5 Score", [0, 1])
A6 = st.selectbox("A6 Score", [0, 1])
A7 = st.selectbox("A7 Score", [0, 1])
A8 = st.selectbox("A8 Score", [0, 1])
A9 = st.selectbox("A9 Score", [0, 1])
A10 = st.selectbox("A10 Score", [0, 1])

age = st.number_input("Age", min_value=1, max_value=100)

gender = st.selectbox("Gender", ['m', 'f'])
jaundice = st.selectbox("Jaundice at birth?", ['yes', 'no'])
austim = st.selectbox("Family history of autism?", ['yes', 'no'])
used_app_before = st.selectbox("Used this screening app before?", ['yes', 'no'])

# Encode categorical values
gender_encoded = 1 if gender == 'm' else 0
jaundice_encoded = 1 if jaundice == 'yes' else 0
austim_encoded = 1 if austim == 'yes' else 0
used_app_encoded = 1 if used_app_before == 'yes' else 0

# Feature order must match training data
feature_names = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'jaundice', 'austim', 'used_app_before'
]

# Create input DataFrame
input_data = pd.DataFrame([[
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
    age, gender_encoded, jaundice_encoded, austim_encoded, used_app_encoded
]], columns=feature_names)

# Predict on button click
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Prediction: **ASD**")
        else:
            st.success("Prediction: **Not ASD**")
    except ValueError as e:
        st.error(f"Prediction failed due to input mismatch: {e}")
