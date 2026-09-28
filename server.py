from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Allow CORS (Cross-Origin Resource Sharing) for frontend-backend communication
from flask_cors import CORS
CORS(app)

MODEL_PATH = r'c:\Users\ACER\file\Speech\speech_recognition_model.h5'
LABEL_ENCODER_PATH = r'c:\Users\ACER\file\Speech\label_encoder.pkl'

# Load the model and label encoder
model = load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request has an audio file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)

    # Save the uploaded file temporarily
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Extract features and make a prediction
    features = extract_features(file_path)
    if features is None:
        return jsonify({"error": "Could not extract features from audio"}), 500

    features = features[np.newaxis, ..., np.newaxis]
    features = np.expand_dims(features, axis=1)  # Reshape for LSTM input
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    # Cleanup uploaded file
    os.remove(file_path)

    return jsonify({"prediction": predicted_label[0]})

if __name__ == '__main__':
    app.run(debug=True)
