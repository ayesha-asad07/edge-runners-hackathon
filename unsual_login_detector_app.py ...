import streamlit as st
import pandas as pd
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the model and encoders
@st.cache(allow_output_mutation=True)
def load_model():
    # Load the trained model
    with open('trained_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load the label encoders for location and device
    with open('label_encoders.json', 'r') as encoders_file:
        label_encoders = json.load(encoders_file)
    
    return model, label_encoders

# Preprocess the user input
def preprocess_input(timestamp, location, device, label_encoders):
    location_encoder = LabelEncoder()
    device_encoder = LabelEncoder()

    # Set the classes_ attribute with proper format (numpy array)
    location_encoder.classes_ = pd.Index(label_encoders['location']).to_numpy()
    device_encoder.classes_ = pd.Index(label_encoders['device']).to_numpy()

    # Encode location and device using loaded encoders
    encoded_location = location_encoder.transform([location])[0]
    encoded_device = device_encoder.transform([device])[0]

    return pd.DataFrame({
        'timestamp': [timestamp],
        'location': [encoded_location],
        'device': [encoded_device]
    })

# Streamlit UI
def main():
    st.title("Login Activity Detection")

    st.write("Enter login details to check if the activity is unusual:")

    # User inputs
    timestamp = st.slider("Timestamp (Hour)", 0, 23, 12)
    location = st.selectbox("Location", ["US", "Europe", "Asia", "Africa", "Australia"])
    device = st.selectbox("Device", ["Desktop", "Mobile", "Tablet"])

    if st.button("Check Activity"):
        # Load the model and encoders
        model, label_encoders = load_model()

        # Preprocess the input
        input_data = preprocess_input(timestamp, location, device, label_encoders)
        
        # Predict if the login is unusual or not
        prediction = model.predict(input_data)[0]

        # Display the result
        if prediction == 1:
            st.error("This login activity is unusual!")
        else:
            st.success("This login activity seems normal.")

if __name__ == "__main__":
    main()
