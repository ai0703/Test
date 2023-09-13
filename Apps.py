%%writefile app.py

import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained models
nb_classifier = joblib.load('naive_bayes_model.pkl')
cnn_model = load_model('cnn_model.h5')

# Define a function to make predictions using each model
def predict_intrusion(input_data):
    # Check if all input values are zero
    if np.all(input_data == 0):
        return "No INTRUSION Detected"

    # Preprocess the input data (e.g., data truncation)
    max_threshold = 10000  # Adjust as needed
    input_data[input_data > max_threshold] = max_threshold

    # Ensure that input_data is of the correct data type (float32)
    input_data = np.array(input_data, dtype=np.float32)

    # Reshape the input data for the CNN model
    input_data_reshaped = input_data.reshape(1, 20, 1)  # Assuming 20 features (excluding Label)

    # Predict using Naive Bayes
    nb_prediction = nb_classifier.predict(input_data)

    # Predict using CNN
    cnn_prediction = cnn_model.predict(input_data_reshaped)

    return nb_prediction, cnn_prediction  # Return both predictions
# Create the Streamlit app
st.title("Intrusion Detection System")

# Create input fields for user inputs (excluding Label)
input_fields = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"
]

user_inputs = {}
for field in input_fields:
    # Ensure all numerical arguments have the same data type (float)
    user_inputs[field] = st.number_input(f"Enter {field}", min_value=0.0, value=0.0, step=0.01, format="%.2f")

# Create a button to trigger prediction
if st.button("Predict"):
    # Prepare user inputs as a NumPy array
    input_data = np.array(list(user_inputs.values())).reshape(1, -1)

    # Make predictions using both models
    predictions = predict_intrusion(input_data)

    # Check if a single string is returned (special case)
    if isinstance(predictions, str):
        st.subheader("Prediction:")
        st.write(predictions)
    else:
        # Unpack predictions when two values are returned
        nb_prediction, cnn_prediction = predictions

        # Determine if it's an intrusion or not based on the predictions
        is_intrusion = nb_prediction[0] == 1 or cnn_prediction[0][0] < 0.5  # Adjust as needed

        # Display the intrusion status
        if is_intrusion:
            st.subheader("Intrusion Status:")
            st.write("INTRUSION Detected")
        else:
            st.subheader("Intrusion Status:")
            st.write("No INTRUSION Detected")

        # Optionally, you can display the individual model predictions as well
        st.subheader("Model Predictions:")
        st.write("Naive Bayes Prediction:", nb_prediction)
        st.write("CNN Prediction:", cnn_prediction)
