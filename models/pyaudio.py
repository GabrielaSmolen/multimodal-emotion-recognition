from pyAudioAnalysis import audioFeatureExtraction, audioTrainTest
from pyAudioAnalysis import ShortTermFeatures
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Load your CSV data
data = pd.read_csv("your_data.csv")  # Replace with your CSV file path

predicted_emotions = []
actual_emotions = []

# Load a pre-trained emotion classification model
model_path = "path_to_pretrained_model"
classifier, mean, std = audioTrainTest.load_model(model_path)

# Iterate through each audio file in the CSV
for index, row in data.iterrows():
    audio_file = row["Audio_File"]
    actual_emotion = row["Actual_Emotion"]

    # Extract audio features
    [features, feature_names] = audioFeatureExtraction.stFeatureExtraction(
        audio_file, 1.0, 1.0, 0.050, 0.050
    )

    # Normalize features
    features_norm = (features - mean) / std

    # Classify emotion
    emotion_label = audioTrainTest.file_classification(features_norm, classifier, "svm")

    predicted_emotions.append(emotion_label)
    actual_emotions.append(actual_emotion)

# Calculate accuracy score and classification report
accuracy = accuracy_score(actual_emotions, predicted_emotions)
classification_rep = classification_report(actual_emotions, predicted_emotions)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)