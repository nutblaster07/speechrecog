import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from difflib import SequenceMatcher
import speech_recognition as sr

DATA_PATH = r'E:\Downloads\test-clean\LibriSpeech\test-clean'

# Paths for the saved model and label encoder
MODEL_PATH = 'speech_recognition_model.h5'
ENCODER_PATH = 'label_encoder.pkl'

# Feature extraction function
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        print(f"Exception: {e}")
        return None

# Function to predict a file's label
def predict(file_path, model, label_encoder):
    print(f"Predicting label for file: {file_path}")
    feature = extract_features(file_path)
    if feature is None:
        print("Feature extraction failed.")
        return None
    feature = feature[np.newaxis, ..., np.newaxis]
    feature = np.expand_dims(feature, axis=1)  # Reshape for LSTM input
    prediction = model.predict(feature)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Speech-to-text function
def speech_to_text(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        print(f"Error in speech recognition for file: {file_path}")
        print(f"Exception: {e}")
        return None

# Function to compare transcription and model prediction
def compare_transcription(file_path, model, label_encoder):
    print(f"Comparing transcription for file: {file_path}")
    transcribed_text = speech_to_text(file_path)
    if transcribed_text is None:
        print("Speech-to-text conversion failed.")
        return None
    
    # Predict label using the model
    predicted_label = predict(file_path, model, label_encoder)
    
    if predicted_label is None:
        print("Model prediction failed.")
        return None
    
    # Comparing transcribed text with the predicted label
    similarity = SequenceMatcher(None, transcribed_text, predicted_label).ratio()
    print(f"Transcribed text: {transcribed_text}")
    print(f"Predicted label: {predicted_label}")
    print(f"Similarity score: {similarity:.2f}")

# Main function for subsequent runs
def main():
    # Check if model and label encoder exist
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        print("Loading saved model and label encoder...")
        model = load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        raise ValueError("Saved model or label encoder not found. Please run the training script first.")
    
    # Test the model with a specific file
    test_file = r'E:\debb.wav'  # Replace with your test audio file path
    compare_transcription(test_file, model, label_encoder)

# Run the script
if __name__ == "__main__":
    main()
