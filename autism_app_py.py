import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set app title and description
st.set_page_config(page_title="Autism Screening App", page_icon="🧩", layout="wide")

# App title
st.title("🧩 Autism Spectrum Disorder (ASD) Screening Tool")
st.markdown("""
This app predicts the likelihood of Autism Spectrum Disorder based on behavioral features 
and demographic information using a machine learning model.
""")

# Sidebar with information
st.sidebar.header("About")
st.sidebar.info("""
This predictive model uses a Random Forest Classifier trained on autism screening data.
The model analyzes responses to behavioral questions and demographic factors to assess ASD risk.
""")

st.sidebar.header("Instructions")
st.sidebar.info("""
1. Answer all the questions below
2. Click the 'Predict' button
3. View your prediction results
""")

# Divide into two columns
col1, col2 = st.columns(2)

with col1:
    # Section header
    st.header("Demographic Information")
    
    # Age input
    age = st.number_input("Age (in years)", min_value=1, max_value=120, value=18)
    
    # Gender input
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    gender_map = {"Male": 1, "Female": 0, "Other": 0.5}
    gender_encoded = gender_map[gender]
    
    # Ethnicity input
    ethnicity_options = [
        "White-European", "Asian", "Middle Eastern", "Black", 
        "South Asian", "Hispanic", "Others", "Mixed", "Latino", 
        "Turkish", "Pasifika", "Others"
    ]
    ethnicity = st.selectbox("Ethnicity", ethnicity_options)
    
    # Country of residence
    country_options = [
        "United States", "United Kingdom", "New Zealand", "Australia",
        "Canada", "Ireland", "Others"
    ]
    country = st.selectbox("Country of Residence", country_options)
    
    # Jaundice history
    jaundice = st.radio("Born with Jaundice?", ["Yes", "No"])
    jaundice_encoded = 1 if jaundice == "Yes" else 0
    
    # Family ASD history
    family_asd = st.radio("Family member with ASD?", ["Yes", "No"])
    family_asd_encoded = 1 if family_asd == "Yes" else 0
    
    # Used app before
    used_app_before = st.radio("Used screening app before?", ["Yes", "No"])
    used_app_before_encoded = 1 if used_app_before == "Yes" else 0

with col2:
    # Section header
    st.header("Behavioral Assessment")
    
    # A1-A10 score questions
    st.markdown("**Please answer the following questions (1-10 scale where 1=Never, 10=Always):**")
    
    a1_score = st.slider("1. I often notice small sounds when others do not", 1, 10, 5)
    a2_score = st.slider("2. I usually concentrate more on the whole picture, rather than the small details", 1, 10, 5)
    a3_score = st.slider("3. I find it easy to do more than one thing at once", 1, 10, 5)
    a4_score = st.slider("4. If there is an interruption, I can switch back to what I was doing very quickly", 1, 10, 5)
    a5_score = st.slider("5. I find it easy to 'read between the lines' when someone is talking to me", 1, 10, 5)
    a6_score = st.slider("6. I know how to tell if someone listening to me is getting bored", 1, 10, 5)
    a7_score = st.slider("7. When I'm reading a story, I find it difficult to work out the characters' intentions", 1, 10, 5)
    a8_score = st.slider("8. I like to collect information about categories of things (e.g., types of cars, birds, trains, plants)", 1, 10, 5)
    a9_score = st.slider("9. I find it easy to work out what someone is thinking or feeling just by looking at their face", 1, 10, 5)
    a10_score = st.slider("10. I find it difficult to work out people's intentions", 1, 10, 5)
    
    # Result (time taken)
    result = st.number_input("Time taken to complete the test (in seconds)", min_value=0, value=300)

# Create a dictionary with all input data
input_data = {
    'A1_Score': a1_score,
    'A2_Score': a2_score,
    'A3_Score': a3_score,
    'A4_Score': a4_score,
    'A5_Score': a5_score,
    'A6_Score': a6_score,
    'A7_Score': a7_score,
    'A8_Score': a8_score,
    'A9_Score': a9_score,
    'A10_Score': a10_score,
    'age': age,
    'gender': gender_encoded,
    'ethnicity': ethnicity,  # Will need encoding
    'jaundice': jaundice_encoded,
    'austim': family_asd_encoded,
    'contry_of_res': country,  # Will need encoding
    'used_app_before': used_app_before_encoded,
    'result': result
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Encoding for categorical variables (simplified for demo)
# In a real app, you would use the same encoding as during model training
input_df['ethnicity'] = input_df['ethnicity'].apply(lambda x: 1 if x == "White-European" else 0)
input_df['contry_of_res'] = input_df['contry_of_res'].apply(lambda x: 1 if x == "United States" else 0)

# Reorder columns to match model training
feature_order = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim', 
    'contry_of_res', 'used_app_before', 'result'
]
input_df = input_df[feature_order]

# Prediction button
if st.button("Predict ASD Likelihood"):
    try:
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error("**Result: High likelihood of ASD**")
            st.warning(f"Probability: {prediction_proba[0][1]*100:.2f}%")
            st.info("""
            This screening result suggests a higher likelihood of Autism Spectrum Disorder. 
            Please consult with a qualified healthcare professional for a comprehensive evaluation.
            """)
        else:
            st.success("**Result: Low likelihood of ASD**")
            st.info(f"Probability: {prediction_proba[0][0]*100:.2f}%")
            st.info("""
            This screening result suggests a lower likelihood of Autism Spectrum Disorder. 
            If you have concerns about development or behavior, consider consulting a healthcare professional.
            """)
            
        # Show feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Key Influencing Factors")
            feature_importance = pd.DataFrame({
                'Feature': feature_order,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(feature_importance.set_index('Feature').head(5))
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool is for screening purposes only and is not a diagnostic instrument. 
A positive screening result does not equate to a diagnosis of ASD. Always consult with a qualified 
healthcare professional for assessment and diagnosis.
""")
