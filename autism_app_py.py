import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Autism Prediction App")
st.write("Please fill out the screening questions:")

# Input form
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
jaundice = st.selectbox("Jaundice at birth", ['yes', 'no'])
austim = st.selectbox("Family History of Autism", ['yes', 'no'])
used_app_before = st.selectbox("Used App Before?", ['yes', 'no'])

# Additional inputs
ethnicity = st.selectbox("Ethnicity", ['White-European', 'Latino', 'Others'])  # Add all classes used in training
contry_of_res = st.selectbox("Country of Residence", ['United States', 'Kenya', 'Others'])  # Add actual options from your training data
result = st.slider("AQ-10 Result Score", 0, 10)
relation = st.selectbox("Who is completing the test?", ['Parent', 'Self', 'Others'])

# Mappings
gender_map = {'m': 1, 'f': 0}
binary_map = {'yes': 1, 'no': 0}
ethnicity_map = {'White-European': 0, 'Latino': 1, 'Others': 2}
country_map = {'United States': 0, 'Kenya': 1, 'Others': 2}
relation_map = {'Parent': 0, 'Self': 1, 'Others': 2}

# Create input DataFrame
input_data = pd.DataFrame([[
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
    age,
    gender_map[gender],
    ethnicity_map[ethnicity],
    binary_map[jaundice],
    binary_map[austim],
    country_map[contry_of_res],
    binary_map[used_app_before],
    result,
    relation_map[relation]
]], columns=[
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'result', 'relation'
])
# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success("Prediction: **ASD**" if prediction[0] == 1 else "Prediction: **Not ASD**")

