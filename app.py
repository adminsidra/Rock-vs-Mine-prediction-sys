import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression

# Set up page configuration
st.set_page_config(
    page_title="Rock vs Mine Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
MODEL_PATH = "model.pkl"
DATA_PATH = "sonar_data.csv"

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #f4f7f6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 6px 10px rgba(0,0,0,0.15);
    }
    .prediction-card-rock {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .prediction-card-mine {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    h1 {
        color: #1f2937;
        font-family: 'Inter', sans-serif;
    }
    h3 {
        color: #4b5563;
        font-family: 'Inter', sans-serif;
    }
    .css-1v0mbdj {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model():
    """
    Loads the trained model if it exists, otherwise trains a new model on the 
    existing dataset and saves it to disk for future use.
    """
    if os.path.exists(MODEL_PATH):
        # Load pre-trained model
        model = joblib.load(MODEL_PATH)
        return model
    else:
        # Check if dataset exists to train
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset '{DATA_PATH}' not found! Please ensure it is in the project folder.")
            return None
        
        # Load dataset
        sonar_data = pd.read_csv(DATA_PATH, header=None)
        
        # Separate features and target
        X = sonar_data.drop(columns=[60])
        Y = sonar_data[60]
        
        # Train model
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X, Y)
        
        # Save model
        joblib.dump(model, MODEL_PATH)
        return model


def main():
    # Header Section
    st.markdown("<h1 style='text-align: center;'>🌊 Rock vs Mine Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Sonar Signal Classification using Machine Learning</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280; font-size: 1.1rem; max-width: 800px; margin: 0 auto;'>This system utilizes a Logistic Regression model to analyze 60 distinct sonar signal frequencies and classify the detected underwater object as either a harmless <b>Rock</b> or a dangerous <b>Mine</b>.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Load Model
    model = load_or_train_model()
    if model is None:
        return

    st.write("### 🎛️ Input Sonar Features")
    st.write("Please enter the 60 sonar numerical feature values below:")

    # Create an input form
    with st.form("prediction_form"):
        # Organize inputs neatly into 4 columns and 15 rows for all 60 features
        columns = st.columns(4)
        input_data = []
        
        # Adding inputs using a loop for a cleaner codebase
        for i in range(60):
            col = columns[i % 4]
            with col:
                # Provide a sensible default of 0.0200 as a placeholder to simulate real inputs
                val = st.number_input(f"Feature {i+1}", value=0.0000, format="%.4f", step=0.0100)
                input_data.append(val)
                
        submit_button = st.form_submit_button(label="🔍 Predict Object")

    # Prediction Logic
    if submit_button:
        # Convert input list to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        
        # Reshape the data for a single instance prediction
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_reshaped)
        
        # Check if the model supports probabilities
        try:
            probabilities = model.predict_proba(input_data_reshaped)[0]
            confidence = max(probabilities) * 100
        except AttributeError:
            confidence = None
        
        st.markdown("---")
        st.write("### 📊 Prediction Result")
        
        # Display Results
        if prediction[0] == 'R':
            st.markdown("""
            <div class="prediction-card-rock">
                <h1 style="color: white; margin: 0;">🪨 Prediction: ROCK</h1>
                <p style="font-size: 1.2rem; margin-top: 10px;">The object is classified as a rock and poses no threat.</p>
            </div>
            """, unsafe_allow_html=True)
            if confidence:
                st.success(f"Confidence Score: **{confidence:.2f}%**")
                st.progress(int(confidence))
                
        else:
            st.markdown("""
            <div class="prediction-card-mine">
                <h1 style="color: white; margin: 0;">💣 Prediction: MINE</h1>
                <p style="font-size: 1.2rem; margin-top: 10px;">Warning! The object is classified as a mine.</p>
            </div>
            """, unsafe_allow_html=True)
            if confidence:
                st.warning(f"Confidence Score: **{confidence:.2f}%**")
                st.progress(int(confidence))


if __name__ == '__main__':
    main()
