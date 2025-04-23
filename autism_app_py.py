import streamlit as st
import pandas as pd
import pickle

# Load model
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'best_model.pkl' is in the same directory.")
    st.stop()

st.title("🧠 Autism Prediction App")
st.write("Answer the following questions to predict the likelihood of Autism Spectrum Disorder (ASD).")

# Screening Questions
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

# NEW REQUIRED FIELDS (based on model)
ethnicity = st.selectbox("Ethnicity", [
    'White-European', 'Latino', 'Others', 'Black', 'Asian',
    'Middle Eastern ', 'South Asian', 'Pasifika', 'Hispanic',
    'Turkish', 'others'
])
country = st.selectbox("Country of Residence", [
    'United States', 'Brazil', 'Spain', 'United Kingdom', 'New Zealand',
    'India', 'Australia', 'Canada', 'Ireland', 'Others'
])
relation = st.selectbox("Who is filling this form?", [
    'Self', 'Parent', 'Relative', 'Health care professional', 'Others'
])
result = st.selectbox("Screening Score Result (Total AQ Score)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Manual encoding (must match training-time encoding)
binary_map = {'yes': 1, 'no': 0}
gender_map = {'m': 1, 'f': 0}
relation_map = {
    'Self': 0, 'Parent': 1, 'Relative': 2, 'Health care professional': 3, 'Others': 4
}
ethnicity_map = {
    'White-European': 0, 'Latino': 1, 'Others': 2, 'Black': 3, 'Asian': 4,
    'Middle Eastern ': 5, 'South Asian': 6, 'Pasifika': 7, 'Hispanic': 8,
    'Turkish': 9, 'others': 10
}
country_map = {
    'United States': 0, 'Brazil': 1, 'Spain': 2, 'United Kingdom': 3, 'New Zealand': 4,
    'India': 5, 'Australia': 6, 'Canada': 7, 'Ireland': 8, 'Others': 9
}

# Create input DataFrame in exact training order
input_data = pd.DataFrame([[
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
    age,
    gender_map[gender],
    binary_map[jaundice],
    binary_map[austim],
    binary_map[used_app_before],
    country_map[country],
    ethnicity_map[ethnicity],
    relation_map[relation],
    result
]], columns=[
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'jaundice', 'austim', 'used_app_before',
    'contry_of_res', 'ethnicity', 'relation', 'result'
])

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.success("Prediction: **ASD**" if prediction[0] == 1 else "Prediction: **Not ASD**")
    except ValueError as e:
        st.error(f"Prediction failed due to input mismatch: {e}")
