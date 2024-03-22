# Classifier.py - Classifies Audio Files

import pandas as pd 
import numpy as np 
import pickle
import librosa 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from collections import Counter

# Module Imports
import utils

class Classifier():
    def __init__(self):
        """
        Constructor for the Classifier class, initializing components needed for music classification.
        This includes loading the trained model and necessary preprocessing tools like the scaler for input normalization
        and a converter for interpreting model output.
        """
        # Initialize the feature extractor to be used for preprocessing audio files
        self.extractor = utils.FeatureExtractor()

        # Load the pre-trained model from specified path
        self.model = keras.models.load_model("./model/model.keras")

        # Load the scaler for input normalization; necessary for preprocessing inputs to match training data distribution
        self.scaler = pickle.load(open('./model/scaler.pkl', 'rb'))

        # Load the class converter to translate model output to human-readable class labels
        self.converter = pickle.load(open('./model/classes.pkl', 'rb'))
    
    def classify_song(self, file_name, offset, duration):
        """
        Classifies a song into predefined categories based on its audio features.

        Parameters:
        - file_name: str, the path to the audio file.
        - offset: int, the starting point in the file to begin analysis (in seconds).
        - duration: int, the duration (in seconds) of the audio clip from the offset to analyze.

        Returns:
        - classification: array, the predicted class label(s) for the song.
        """
        # Extract features from the song using the FeatureExtractor instance
        music_features = self.extractor.process_music(file_name, offset, duration)
        
        # Normalize the extracted features using the loaded scaler
        scaled_input = self.scaler.transform(np.array(music_features, dtype=float))
        
        # Use the loaded model to predict the class of the song based on the scaled features
        raw_classification = self.model.predict(scaled_input)
        
        # Convert the model's prediction (numerical) back to the original class labels
        classification = self.converter.inverse_transform(np.argmax(raw_classification, axis=1))
        
        return classification
    
    def enhanced_classify(self, file_name):
        """
        Classifies a song into predefined categories based on its audio features by splitting it into 3-second segments.

        Parameters:
        - file_name: str, the path to the audio file.

        Returns:
        - predicted_genre: str, the predicted genre of the song based on the majority of segment classifications.
        """
        # Load the audio file
        y, sr = librosa.load(file_name)

        # Calculate the total duration of the audio file in seconds
        duration = librosa.get_duration(y=y, sr=sr)

        # Initialize a Counter to store the classification counts for each segment
        classification_counts = Counter()

        # Split the audio into 3-second segments and classify each segment
        for offset in range(0, int(duration), 6):
            # Extract features from the current segment using the FeatureExtractor instance
            music_features = self.extractor.process_music(file_name, offset, 3)

            # Normalize the extracted features using the loaded scaler
            scaled_input = self.scaler.transform(np.array(music_features, dtype=float))

            # Use the loaded model to predict the class of the segment based on the scaled features
            raw_classification = self.model.predict(scaled_input)

            # Convert the model's prediction (numerical) back to the original class labels
            classification = self.converter.inverse_transform(np.argmax(raw_classification, axis=1))

            # Increment the count for the predicted genre in the classification_counts Counter
            classification_counts[classification[0]] += 1

        # Determine the predicted genre based on the majority of segment classifications
        predicted_genre = classification_counts.most_common(1)[0][0]

        print("Other Detected Sub-Genres Include: ", classification_counts)
        print("Main Genre Detected: ", predicted_genre)
        return predicted_genre
