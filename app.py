# Import the necessary libraries
import streamlit as st
import joblib
import json
import numpy as np

# Load the model
model = joblib.load('personality_classifier_xgb.pkl')

# Load the label mappings
with open('label_mappings.json', 'r') as file:
    label_mappings = json.load(file)

# Load personality type descriptions
with open('personality_descriptions.json', 'r') as file:
    personality_descriptions = json.load(file)


# Convert the label mappings to use in predictions
gender_mapping = label_mappings['Gender']
interest_mapping = label_mappings['Interest']
personality_mapping = {v: k for k, v in label_mappings['Personality'].items()}

class PersonalityPredictor:
    def __init__(self):
        self.ini_session_state()

    def ini_session_state(self):
        """Initialize session states for all input fields."""
        if 'step' not in st.session_state:
            st.session_state.step = 0  # Start at step 0 for introduction
        if 'age' not in st.session_state:
            st.session_state.age = 21
        if 'gender' not in st.session_state:
            st.session_state.gender = 1
        if 'education' not in st.session_state:
            st.session_state.education = 0
        if 'introversion_score' not in st.session_state:
            st.session_state.introversion_score = 5.0
        if 'sensing_score' not in st.session_state:
            st.session_state.sensing_score = 5.0
        if 'thinking_score' not in st.session_state:
            st.session_state.thinking_score = 5.0
        if 'judging_score' not in st.session_state:
            st.session_state.judging_score = 5.0
        if 'interest' not in st.session_state:
            st.session_state.interest = 0
        if 'prediction_done' not in st.session_state:
            st.session_state.prediction_done = False

    def collect_inputs(self):
        """Prepare the input data for model prediction."""
        input_data = np.array([[st.session_state.age, st.session_state.gender, st.session_state.education,
                                 st.session_state.introversion_score, st.session_state.sensing_score,
                                 st.session_state.thinking_score, st.session_state.judging_score,
                                 st.session_state.interest]])
        return input_data

    def predict(self):
        """Predict the personality type."""
        input_data = self.collect_inputs()
        prediction = model.predict(input_data)
        return personality_mapping[prediction[0]]

    def scale_score(self, score_list):
        """Scale the combined scores of three questions to a 0-10 range."""
        return sum(score_list) * (9.9 / 9.9)
    
    def take_input(self):
        """Show input fields based on the current step with more intuitive prompts."""

        if st.session_state.step == 1:
            st.session_state.age = st.number_input("How old are you?", min_value=18, max_value=57, value=21, step=1)

        elif st.session_state.step == 2:
            gender = st.radio("What is your gender?", options=["Male", "Female"], horizontal=True)
            st.session_state.gender = 1 if gender == "Male" else 0

        elif st.session_state.step == 3:
            education = st.radio("What is your highest level of education?", 
                                options=["Undergraduate/High School/Uneducated", "Graduate or Higher"], 
                                horizontal=True)
            st.session_state.education = 1 if education == "Graduate or Higher" else 0

        # Introversion questions in separate steps | High implies Extroversion 
        elif st.session_state.step == 4:
            st.session_state.introversion_q1 = st.slider("Do you enjoy spending time in large gatherings over intimate settings?", 
                                                        min_value=0.0, max_value=1.0, value=0.5) * 3.3
        elif st.session_state.step == 5:
            st.session_state.introversion_q2 = st.slider("Do you feel energized by socializing with many people?", 
                                                        min_value=0.0, max_value=1.0, value=0.5) * 3.3
        elif st.session_state.step == 6:
            st.session_state.introversion_q3 = st.slider("Do you prefer engaging in lively activities over quiet, reflective time alone?", 
                                                        min_value=0.0, max_value=1.0, value=0.5) * 3.3

            st.session_state.introversion_score = self.scale_score([
                st.session_state.introversion_q1, 
                st.session_state.introversion_q2, 
                st.session_state.introversion_q3])

        # Sensing questions in separate steps
        elif st.session_state.step == 7:
            st.session_state.sensing_q1 = st.slider("Do you prefer dealing with facts over abstract concepts?", 
                                                    min_value=0.0, max_value=1.0, value=0.5) * 3.3
        elif st.session_state.step == 8:
            st.session_state.sensing_q2 = st.slider("Do you tend to notice details others might miss?", 
                                                    min_value=0.0, max_value=1.0, value=0.5) * 3.3
        elif st.session_state.step == 9:
            st.session_state.sensing_q3 = st.slider("Do you value practical solutions over theoretical ideas?", 
                                                    min_value=0.0, max_value=1.0, value=0.5) * 3.3
            st.session_state.sensing_score = self.scale_score([
                st.session_state.sensing_q1, 
                st.session_state.sensing_q2, 
                st.session_state.sensing_q3])

        # Thinking questions in separate steps
        elif st.session_state.step == 10:
            st.session_state.thinking_q1 = st.slider("Do you prioritize logical reasoning when making decisions?", 
                                                     min_value=0.0, max_value=1.0, value=0.5) * 3.3
        elif st.session_state.step == 11:
            st.session_state.thinking_q2 = st.slider("Do you find yourself focusing more on objective outcomes than personal emotions?", 
                                                     min_value=0.0, max_value=1.0, value=0.5) * 3.3
        elif st.session_state.step == 12:
            st.session_state.thinking_q3 = st.slider("Do you value efficiency over maintaining harmony?", 
                                                     min_value=0.0, max_value=1.0, value=0.5) * 3.3
            st.session_state.thinking_score = self.scale_score([
                st.session_state.thinking_q1, 
                st.session_state.thinking_q2, 
                st.session_state.thinking_q3])

        # Judging questions in separate steps
        elif st.session_state.step == 13:
            st.session_state.judging_q1 = st.slider("Do you prefer planning things ahead of time?", 
                                                    min_value=0.0, max_value=1.0, value=0.5) * 3.3
        elif st.session_state.step == 14:
            st.session_state.judging_q2 = st.slider("Do you feel most comfortable when following a structured routine?", 
                                                    min_value=0.0, max_value=1.0, value=0.5) * 3.3
        elif st.session_state.step == 15:
            st.session_state.judging_q3 = st.slider("Do you find it satisfying to complete tasks before moving on?", 
                                                    min_value=0.0, max_value=1.0, value=0.5) * 3.3
            st.session_state.judging_score = self.scale_score([
                st.session_state.judging_q1, 
                st.session_state.judging_q2, 
                st.session_state.judging_q3])

        elif st.session_state.step == 16:
            interest = st.selectbox("What is your primary area of interest?", 
                                    options=["Arts", "Sports", "Technology", "Others", "Unknown"])
            st.session_state.interest = interest_mapping[interest]

    def next_step(self):
        """Go to the next step."""
        if st.session_state.step < 16:
            st.session_state.step += 1

    def previous_step(self):
        """Go to the previous step."""
        if st.session_state.step > 1:
            st.session_state.step -= 1

def show_intro():
    """Displays the introduction on Step 0."""
    st.title("MBTI Personality Type Predictor")
    st.write("""
    The MBTI (Myers-Briggs Type Indicator) is a self-report questionnaire indicating differing psychological 
    preferences in how people perceive the world and make decisions. The MBTI personality test classifies 
    individuals into 16 distinct personality types.

    This app will predict your MBTI personality type based on your inputs. 
    Click 'Start' to begin the test.
    """)
    st.button("Start", type="primary", on_click=start_test)
    
def start_test():
    st.session_state.step = 1
    
def restart_test():
    st.session_state.step = 0
    st.session_state.prediction_done = False

def main():
    predictor = PersonalityPredictor()

    # Step 0: Introduction
    if st.session_state.step == 0:
        show_intro()
    elif st.session_state.prediction_done:
        st.title("Your Predicted Personality Type")
        input_data = predictor.collect_inputs()
        st.write(f"**Age**: {input_data[0][0]}, **Gender**: {input_data[0][1]}, **Education**: {input_data[0][2]}, **Extroversion**: {input_data[0][3]}, **Sensing**: {input_data[0][4]}, **Thinking**: {input_data[0][5]}, **Judging**: {input_data[0][6]}, **Interest**: {input_data[0][7]}")
        
        st.markdown(f"<h2 style='color: blue;'>{st.session_state.prediction}</h2>", unsafe_allow_html=True)
        
        # Display personality description
        description = personality_descriptions.get(st.session_state.prediction, "No description available.")
        st.write(description)
            
        st.write("Thank you for completing the test!")
        st.button("Restart Test", on_click=restart_test)
    else:
        # Step 1-8: Collecting inputs
        predictor.take_input()

        # Navigation buttons
        col1, col2, col3 = st.columns(3, gap="small")
        
        with col1:
            if st.button("Previous", on_click=predictor.previous_step, disabled=(st.session_state.step == 1)):
                pass

        with col2:
            if st.button("Next", on_click=predictor.next_step, disabled=(st.session_state.step == 16)):
                pass        

        with col3:
            if st.session_state.step == 16:
                if st.button("Predict Personality Type", type="primary"):
                    personality_type = predictor.predict()
                    st.session_state.prediction = personality_type
                    st.session_state.prediction_done = True
                    st.rerun()

if __name__ == "__main__":
    main()
