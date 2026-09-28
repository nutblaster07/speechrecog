import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import speech_recognition as sr
import pickle
from difflib import SequenceMatcher

DATA_PATH = r'E:\Downloads\test-clean\LibriSpeech\test-clean'
MODEL_SAVE_PATH = 'speech_recognition_model.h5'
LABEL_ENCODER_SAVE_PATH = 'label_encoder.pkl'
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

def load_data(data_path):
    print(f"Loading data from: {data_path}")
    features = []
    labels = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(os.path.basename(root))
    print(f"Data loading complete. Number of samples: {len(features)}")
    return np.array(features), np.array(labels)

print("Starting data loading...")
features, labels = load_data(DATA_PATH)

if len(features) == 0 or len(labels) == 0:
    raise ValueError("No data found. Please check the DATA_PATH and ensure it contains .flac files.")

print("Data loaded. Encoding labels...")
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
if os.path.exists(LABEL_ENCODER_SAVE_PATH):
    print("Loading the saved label encoder...")
    with open(LABEL_ENCODER_SAVE_PATH, 'rb') as file:
        le = pickle.load(file)
    print("Label encoder loaded successfully.")
else:
    print("Encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    with open(LABEL_ENCODER_SAVE_PATH, 'wb') as file:
        pickle.dump(le, file)
    print(f"Label encoder saved to {LABEL_ENCODER_SAVE_PATH}")

print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

print("Reshaping data for RNN input...")
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Reshaping for LSTM: [samples, time_steps, features]
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

print("Building and compiling the RNN model...")
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(np.unique(labels)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training the model...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

print("Plotting training history...")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()

def predict(file_path):
    print(f"Predicting label for file: {file_path}")
    feature = extract_features(file_path)
    if feature is None:
        print("Feature extraction failed.")
        return None
    feature = feature[np.newaxis, ..., np.newaxis]
    feature = np.expand_dims(feature, axis=1)  # Reshape for LSTM input
    prediction = model.predict(feature)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

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

def compare_transcription(file_path):
    print(f"Comparing transcription for file: {file_path}")
    transcribed_text = speech_to_text(file_path)
    if transcribed_text is None:
        print("Speech-to-text conversion failed.")
        return None
    
    # Predict label using the model
    predicted_label = predict(file_path)
    
    if predicted_label is None:
        print("Model prediction failed.")
        return None
    
    # Comparing transcribed text with the predicted label
    similarity = SequenceMatcher(None, transcribed_text, predicted_label).ratio()
    print(f"Transcribed text: {transcribed_text}")
    print(f"Predicted label: {predicted_label}")
    print(f"Similarity score: {similarity:.2f}")

# Replace this with a specific file path you want to test
compare_transcription(r'E:\te3.wav')
