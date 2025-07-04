import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

from absl import logging
logging.set_verbosity(logging.ERROR)

import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize # type: ignore

# Load the model once and reuse it
model = tf.keras.models.load_model("Trained_model.h5")

# Preprocess the audio file
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Chunk configuration
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    
    # Convert to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    # Number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    # Iterate through chunks
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        
        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        
        data.append(mel_spectrogram)
    
    return np.array(data)

# Model Prediction
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    
    return max_elements[0]
