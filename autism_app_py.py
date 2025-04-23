import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("Autism Prediction App")

# Input fields — replace/add based on your model's input features
age = st.slider("Age", 1, 100, 25)
jaundice = st.selectbox("History of jaundice?", ["Yes", "No"])
family_history = st.selectbox("Family history of autism?", ["Yes", "No"])
score = st.slider("Screening Test Score (0-10)", 0, 10, 5)

# Map categorical inputs to numerical values
jaundice_val = 1 if jaundice == "Yes" else 0
family_history_val = 1 if family_history == "Yes" else 0

# Combine all inputs into a single array
input_data = np.array([[age, jaundice_val, family_history_val, score]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("Prediction: At Risk of Autism")
    else:
        st.success("Prediction: Not At Risk of Autism")
