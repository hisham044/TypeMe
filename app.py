## Streamlit app to predict the MBTI personality type based on user inputs

# Import the necessary libraries
import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('personality_classifier_xgb.pkl')

# Load the label mappings
with open('label_mappings.json', 'r') as file:
    label_mappings = json.load(file)

# Convert the label mappings to use in predictions
gender_mapping = label_mappings['Gender']
interest_mapping = label_mappings['Interest']
personality_mapping = {v: k for k, v in label_mappings['Personality'].items()}

class PersonalityPredictor:
    def __init__(self):
        self.age = None
        self.gender = None
        self.education = None
        self.introversion_score = None
        self.sensing_score = None
        self.thinking_score = None
        self.judging_score = None
        self.interest = None

    def collect_inputs(self):
        # Collect inputs from user
        self.age = st.session_state.get('age', 0)
        self.gender = st.session_state.get('gender', None)
        self.education = st.session_state.get('education', None)
        self.introversion_score = st.session_state.get('introversion_score', 0)
        self.sensing_score = st.session_state.get('sensing_score', 0)
        self.thinking_score = st.session_state.get('thinking_score', 0)
        self.judging_score = st.session_state.get('judging_score', 0)
        self.interest = st.session_state.get('interest', None)

    def predict(self):
        # Prepare input for the model
        input_data = np.array([[self.age, self.gender, self.education,
                                 self.introversion_score, self.sensing_score,
                                 self.thinking_score, self.judging_score,
                                 self.interest]])
        
        # Predict the personality type
        prediction = model.predict(input_data)
        return personality_mapping[prediction[0]]

def main():
    st.title("MBTI Personality Type Predictor")
    st.session_state.setdefault('started', False)
    
    if not st.session_state['started']:
        st.write("Welcome to the MBTI Personality Type Prediction App!")
        if st.button("Start Personality Type Test"):
            st.session_state['started'] = True

    if st.session_state['started']:
        predictor = PersonalityPredictor()
        
        # Input fields
        age = st.number_input("Age (18-57)", min_value=18, max_value=57, value=21)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        education = st.selectbox("Education Level", options=["Undergraduate/High School/Uneducated", "Graduate or Higher"])
        introversion_score = st.slider("Introversion Score (0-10)", min_value=0.0, max_value=10.0, value=5.0)
        sensing_score = st.slider("Sensing Score (0-10)", min_value=0.0, max_value=10.0, value=5.0)
        thinking_score = st.slider("Thinking Score (0-10)", min_value=0.0, max_value=10.0, value=5.0)
        judging_score = st.slider("Judging Score (0-10)", min_value=0.0, max_value=10.0, value=5.0)
        interest = st.selectbox("Primary Area of Interest", options=["Arts", "Sports", "Technology", "Others"])
        
        # Convert inputs to their respective encoded values
        st.session_state['age'] = age
        st.session_state['gender'] = gender_mapping[gender]
        st.session_state['education'] = 1 if education == "Graduate or Higher" else 0
        st.session_state['introversion_score'] = introversion_score
        st.session_state['sensing_score'] = sensing_score
        st.session_state['thinking_score'] = thinking_score
        st.session_state['judging_score'] = judging_score
        st.session_state['interest'] = interest_mapping[interest]
        
        # Button to make a prediction
        if st.button("Predict Personality Type"):
            predictor.collect_inputs()
            personality_type = predictor.predict()
            st.write(f"Predicted Personality Type: {personality_type}")

if __name__ == "__main__":
    main()
