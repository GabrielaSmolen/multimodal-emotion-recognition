import neattext.functions as nfx
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense
import nltk
from sklearn.model_selection import StratifiedKFold
from googletrans import Translator
import random


translator = Translator()

nltk.download('punkt')


# Load your original data
data = pd.read_csv("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/text_data.csv")

# Define the target number of samples for the "hap" emotion
target_samples = 1500


# Function to perform back-translation
def back_translate(text):
    try:
        translation = translator.translate(text, src="en", dest="fr")
        back_translation = translator.translate(translation.text, src="fr", dest="en")
        return back_translation.text
    except:
        return text


augmented_data = []
for emotion in data['emotion'].unique():
    emotion_data = data[data['emotion'] == emotion]

    augmented_sentences = []
    for _ in range(target_samples):
        random_index = random.choice(emotion_data.index)
        original_sentence = emotion_data.loc[random_index, 'sentence']
        back_translated_sentence = back_translate(original_sentence)
        augmented_sentences.append(back_translated_sentence)

    augmented_emotion_data = pd.DataFrame({'emotion': emotion, 'sentence': augmented_sentences})
    augmented_data.append(augmented_emotion_data)

combined_data = pd.concat([data, augmented_data], ignore_index=True)

combined_data['Clean_Text'] = combined_data['sentence'].apply(nfx.remove_userhandles)
combined_data['Clean_Text'] = combined_data['Clean_Text'].apply(nfx.remove_stopwords)

combined_data['emotion'].value_counts()

# Convert sentences to CountVectorizer format
cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))
data_cv = cv.fit_transform(combined_data['Clean_Text']).toarray()

# Encode the labels
label_encoder = preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(combined_data['emotion'])


# Define a function to create your Keras model
def create_model():
    model = Sequential()
    model.add(Dense(12, input_shape=(data_cv.shape[1],), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Initialize cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []

# Perform cross-validation
for train_idx, test_idx in cv.split(data_cv, encoded_labels):
    X_train, X_test = data_cv[train_idx], data_cv[test_idx]
    y_train, y_test = encoded_labels[train_idx], encoded_labels[test_idx]

    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=1)

    # Make predictions and calculate accuracy
    y_pred_probs = model.predict(X_test)
    y_test_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate accuracy on test data
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # Generate classification report for test data
    test_classification_rep = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_)
    print("Test Classification Report:\n", test_classification_rep)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert predicted probabilities to class labels
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Print cross-validation results
print("Cross-Validation Scores:", accuracy_scores)
print("Mean CV Score:", np.mean(accuracy_scores))
print("Standard Deviation:", np.std(accuracy_scores))
