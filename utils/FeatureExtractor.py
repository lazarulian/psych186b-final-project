# FeatureExtractor.py - Will extract relevant features from raw audio tracks to input into the model. 

import pandas as pd 
import numpy as np 
import sys 
import pickle 
import librosa

# Feature Extraction

def calculate_mean_var(series, res):
    """
    Calculate the mean and variance of a numpy array (series), and append these values to the provided list (res).

    Parameters:
    - series: numpy array, the data series from which to calculate mean and variance.
    - res: list, the list to which the mean and variance should be appended.

    Returns:
    - res: list, the updated list with mean and variance appended.
    """

    mean = np.average(series)
    var = np.var(series)

    res.append(mean)
    res.append(var)
    return res
    

class FeatureExtractor():
    def __init__(self):
        """
        Constructor for the FeatureExtractor class, initializing attributes to store audio data, sampling rate, extracted features, and formatted output.
        """
        # Attributes for storing audio data, sampling rate, and features
        self.track_name = None
        self.data = None
        self.sr = None
        self.features = None
        self.output = None

        # Header for the Output Dataframe
        self.headers = ["length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var", 
            "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", 
            "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", 
            "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo", 
            "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", "mfcc3_mean", "mfcc3_var", 
            "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", "mfcc6_var", 
            "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", 
            "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", 
            "mfcc13_mean", "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var", 
            "mfcc16_mean", "mfcc16_var", "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var", 
            "mfcc19_mean", "mfcc19_var", "mfcc20_mean", "mfcc20_var" ]

    def __load_music(self, file_name, offset=0, duration=3):
        """
        Private method to load an audio file using librosa.

        Parameters:
        - file_name: str, the path to the audio file to be loaded.
        - offset: int, offset into the track
        - duration: int, duration of track to load

        Sets the `track_name`, `data`, and `sr` (sampling rate) attributes.
        """

        try:
            self.track_name = file_name.replace('.wav', '')
            self.data, self.sr = librosa.load(file_name, duration=duration, offset=offset)
        except Exception as e:
            print(f"Failed to load music track: {e}")

    def __extract_features(self):
        """
        Private method to extract audio features using librosa.

        Returns:
        - features: list, the extracted features.
        """

        file_name = self.track_name
        data = self.data
        sr = self.sr
        
        features = []

        # Extracting Relevant Features for Model
        try: 
            length = len(data); features.append(length)
            chroma_stft = librosa.feature.chroma_stft(y=data, sr=sr); calculate_mean_var(chroma_stft, features)
            rms = librosa.feature.rms(y=data); calculate_mean_var(rms, features)
            spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr); calculate_mean_var(spectral_centroid, features)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr); calculate_mean_var(spectral_bandwidth, features)
            rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr); calculate_mean_var(rolloff, features)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=data); calculate_mean_var(zero_crossing_rate, features)
            harmonic = librosa.effects.harmonic(y=data); calculate_mean_var(harmonic, features)
            percussive = librosa.effects.percussive(y=data); calculate_mean_var(percussive, features)
            tempo = librosa.feature.tempo(y=data); features.append(tempo[0])
            mfcc = librosa.feature.mfcc(y=data, sr=sr);
            for i in range(0, 20):
                temp = mfcc[i]
                calculate_mean_var(temp, features)
        
        except Exception as e:
            print(f"Failed to extract features: {e}")

        return features
    
    def __format_output(self):
        """
        Private method to format the extracted features into a pandas DataFrame.

        Returns:
        - df: DataFrame, the formatted output with features as columns.
        """
        try:
            data = {header: [value] for header, value in zip(self.headers, self.features)}    
            df = pd.DataFrame(data)
        
        except Exception as e:
            print(f"Failed to return output dataframe: {e}")

        return df
    
    def process_music(self, file_name, offset=0, duration=3):
        """
        Public method to process an audio file, extract features, and format the output.

        Parameters:
        - file_name: str, the path to the audio file.
        - offset: int, offset into the track
        - duration: int, duration of track to load

        Returns:
        - output: DataFrame, the extracted features formatted as a DataFrame.
        """
        self.__load_music(file_name, offset, duration)
        self.features = self.__extract_features()
        self.output = self.__format_output()
        print(self.output)
        return self.output