## Classifier API

The `Classifier` class provides functionality for classifying audio files into predefined music genres. It uses a pre-trained machine learning model to analyze audio features and make predictions.

### Initialization

To create an instance of the `Classifier` class, simply call the constructor:

```python
from Classifier import Classifier

classifier = Classifier()
```

The constructor initializes the necessary components for music classification, including loading the pre-trained model, scaler, and class converter.

### Methods

#### `classify_song(file_name, offset, duration)`

Classifies a song into predefined categories based on its audio features.

Parameters:

- `file_name` (str): The path to the audio file.
- `offset` (int): The starting point in the file to begin analysis (in seconds).
- `duration` (int): The duration (in seconds) of the audio clip from the offset to analyze.

Returns:

- `classification` (array): The predicted class label(s) for the song.

#### `enhanced_classify(file_name)`

Classifies a song into predefined categories based on its audio features by splitting it into 3-second segments.

Parameters:

- `file_name` (str): The path to the audio file.

Returns:

- `predicted_genre` (str): The predicted genre of the song based on the majority of segment classifications.

### Usage

To use the `Classifier` class, follow these steps:

1. Create an instance of the `Classifier` class:

   ```python
   classifier = Classifier()
   ```

2. Call the desired classification method with the appropriate parameters:

   ```python
   # Classify a song using a specific offset and duration
   classification = classifier.classify_song("path/to/song.mp3", offset=60, duration=3)

   # Classify a song using the enhanced classification method
   predicted_genre = classifier.enhanced_classify("path/to/song.mp3")
   ```

3. The `classify_song` method returns an array of predicted class labels for the specified audio segment, while the `enhanced_classify` method returns the predicted genre based on the majority of segment classifications.

### Example

Here's an example of how to use the `Classifier` class:

```python
# Initialize the classifier
classifier = Classifier()

# Specify the song files and parameters
song_files = [
    "./data/Ex-Factor_LaurynHill.mp3",
    "./data/CaliforniaLove_Tupac.wav",
    "ASongofIceandFire_GameOfThrones.wav"
]
clip_duration = 3
song_offset = 60

# Classify the songs
for song_file in song_files:
    classification = classifier.classify_song(song_file, song_offset, clip_duration)
    print(f"Classification for {song_file}: {classification}")

    predicted_genre = classifier.enhanced_classify(song_file)
    print(f"Predicted genre for {song_file}: {predicted_genre}")
```

In this example, we initialize the `Classifier` instance, specify the song files and parameters, and then use the `classify_song` and `enhanced_classify` methods to classify each song. The results are printed to the console.

Note: Make sure to set up your environment according to the developer instructions before using the `Classifier` class.

## Developer Instructions

**Initializing Virtual Environment**: The virtual environment will allow us to manage dependencies together and eventually deploy our project.

```bash
# Installing Virtual Environment
python3 -m venv venv

# Activating Virtual Environment
source venv/bin/activate

# Installing Requirements
pip install -r requirements.txt

# Saving New Packages Added (Whenever you install a new requirement into virtual environment)
pip freeze > requirements.txt
```

**Downloading the Data:** You need to download the data so that the model can train properly. Use this [link](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection) to download the file. Please place the contents in a folder named `data` in the root directory of the project.

## Project Overview

- **Objective**: To create a machine learning model capable of classifying music samples into different genres based on audio signals.
- **Motivation**: Automating music classification to streamline the selection process, uncover trends, and facilitate easier discovery of music.
- **Dataset**: Utilization of the GTZAN Genre Classification dataset, which includes 1,000 audio tracks across 10 genres, each 30 seconds long.

### Training

- **Data Preparation**:
  - Import necessary libraries and read the dataset.
  - Drop unnecessary columns (e.g., 'filename') and focus on relevant features.
- **Feature Extraction**:
  - Use Librosa for audio analysis and feature extraction, including tempo, chroma, spectral features, and zero-crossing rates.
  - Preprocess data by encoding categorical labels and scaling features.
- **Model Architecture**:
  - Employ Convolutional Neural Networks (CNN) due to their effectiveness in handling image (spectrogram) representations of audio signals.
  - Configure the model with layers, activation functions (RELU, softmax), and the Adam optimizer.

### Testing

- **Model Evaluation**:
  - Train the model using a split of training and testing data, monitoring for overfitting.
  - Evaluate model performance using accuracy metrics and adjust parameters as necessary to improve results.
  - Evaluate model based off of various edge cases listed below
    - Reversed audio file
    - Audio files that are noisy
    - Audio files that filter out high frequency and low frequency
    - Audio files that are indicative of multiple genres

### Platform Building

- **Deployment**:
  - Integrate the trained model into a user-friendly platform for easy music genre classification.
  - Ensure the platform can accept new music data, process it through the model, and display the predicted genre.

### Edge Cases

- **Handling Music in Various Formats**
  - Handling music that is played backwards into the model.
  - Handling music that is blended with other clips.
  - Handling music that is noisy in terms of their spectrograph, maybe with background noise.
- **Handling Diverse Data**:
  - Address challenges with audio files that may belong to multiple genres or none of the predefined categories.
- **Improving Model Robustness**:
  - Explore strategies for dealing with low-quality audio samples or tracks with significant background noise.
- **Adapting to New Genres**:
  - Consider mechanisms for updating the model as new genres emerge or existing ones evolve.
