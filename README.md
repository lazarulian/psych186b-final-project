## Use the Classifier
To try out the classifier, set your environment up using the developer instructions, then hop into a notebook to use the `Classifier` module. I already set one up titled `Predict.ipynb`. Within the playground, you can call and use the classifier as follows. 

```python3
# Initialize Parameters
song_files = ["./data/Ex-Factor_LaurynHill.mp3", "./data/CaliforniaLove_Tupac.wav", "ASongofIceandFire_GameOfThrones.wav"]
clip_duration = 3
song_offset = 60

# Classify
classifier.classify_song(song_files[0], song_offset, clip_duration)
classifier.classify_song(song_files[1], song_offset, clip_duration)
classifier.classify_song(song_files[2], song_offset, clip_duration)
```

```python3
# Outputs

1/1 [==============================] - 0s 72ms/step
Classification:  pop
array(['pop'], dtype=object)

1/1 [==============================] - 0s 81ms/step
Classification:  hiphop
array(['hiphop'], dtype=object)

1/1 [==============================] - 0s 69ms/step
Classification:  classical
array(['classical'], dtype=object)
```

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

## Literature

- [Convolutional Neural Networks Approach for Music Genre Classification](https://ieeexplore.ieee.org/document/9394067)
  - Exploring the workings of how CNN's operate using our dataset. This is more of a scientific paper.
- [Music Genre Classification Using CNN](https://www.clairvoyant.ai/blog/music-genre-classification-using-cnn)
  - Simple example of classification using CNN

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
