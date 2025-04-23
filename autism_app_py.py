import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'best_model.pkl' is in the same directory.")
    st.stop()

# Extract expected features from model
try:
    expected_columns = list(model.feature_names_in_)
except AttributeError:
    st.error("Model is missing 'feature_names_in_'. Make sure it was trained with a DataFrame.")
    st.stop()

st.title("🧠 Autism Prediction App")
st.write("Fill in the screening questions to get a prediction.")

# Input values
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
used_app_before = st.selectbox("Used this app before?", ['yes', 'no'])

ethnicity = st.selectbox("Ethnicity", [
    'White-European', 'Latino', 'Others', 'Black', 'Asian',
    'Middle Eastern ', 'South Asian', 'Pasifika', 'Hispanic', 'Turkish', 'others'
])
country = st.selectbox("Country of Residence", [
    'United States', 'Brazil', 'Spain', 'United Kingdom', 'New Zealand',
    'India', 'Australia', 'Canada', 'Ireland', 'Others'
])
relation = st.selectbox("Relation to individual", [
    'Self', 'Parent', 'Relative', 'Health care professional', 'Others'
])
result = st.selectbox("AQ Score Result (0-10)", list(range(0, 11)))

# Mappings (must match model training)
binary_map = {'yes': 1, 'no': 0}
gender_map = {'m': 1, 'f': 0}
ethnicity_map = {
    'White-European': 0, 'Latino': 1, 'Others': 2, 'Black': 3, 'Asian': 4,
    'Middle Eastern ': 5, 'South Asian': 6, 'Pasifika': 7, 'Hispanic': 8,
    'Turkish': 9, 'others': 10
}
country_map = {
    'United States': 0, 'Brazil': 1, 'Spain': 2, 'United Kingdom': 3, 'New Zealand': 4,
    'India': 5, 'Australia': 6, 'Canada': 7, 'Ireland': 8, 'Others': 9
}
relation_map = {
    'Self': 0, 'Parent': 1, 'Relative': 2, 'Health care professional': 3, 'Others': 4
}

# Create raw input dictionary
raw_input = {
    'A1_Score': A1, 'A2_Score': A2, 'A3_Score': A3, 'A4_Score': A4, 'A5_Score': A5,
    'A6_Score': A6, 'A7_Score': A7, 'A8_Score': A8, 'A9_Score': A9, 'A10_Score': A10,
    'age': age,
    'gender': gender_map[gender],
    'jaundice': binary_map[jaundice],
    'austim': binary_map[austim],
    'used_app_before': binary_map[used_app_before],
    'contry_of_res': country_map[country],
    'ethnicity': ethnicity_map[ethnicity],
    'relation': relation_map[relation],
    'result': result
}

# Build DataFrame in exact model order
try:
    input_data = pd.DataFrame([[raw_input[col] for col in expected_columns]], columns=expected_columns)
except KeyError as e:
    st.error(f"Missing expected input column: {e}")
    st.stop()

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.success("Prediction: **ASD**" if prediction[0] == 1 else "Prediction: **Not ASD**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
