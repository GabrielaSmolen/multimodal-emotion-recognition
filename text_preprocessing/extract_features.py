import random
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec


data = pd.read_csv("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/text_data.csv")

# Define the target number of samples for each emotion
target_samples = 2000

# Dictionary to store augmented data
augmented_data = {"emotion": [], "sentence": []}

# Iterate through each emotion
for emotion in data["emotion"].unique():
    emotion_data = data[data["emotion"] == emotion]
    existing_samples = len(emotion_data)
    samples_needed = target_samples - existing_samples

    # If more samples are needed for this emotion
    if samples_needed > 0:
        samples_to_augment = random.choices(emotion_data["sentence"].tolist(), k=samples_needed)
        augmented_data["emotion"].extend([emotion] * samples_needed)
        augmented_data["sentence"].extend(samples_to_augment)

# Create a DataFrame from the augmented data
augmented_df = pd.DataFrame(augmented_data)

# Combine original data and augmented data
combined_data = pd.concat([data, augmented_df], ignore_index=True)

# Shuffle the combined data
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(combined_data['emotion'].value_counts())

dir(nfx)

combined_data['Clean_Text'] = combined_data['Text line'].apply(nfx.remove_userhandles)

combined_data['Clean_Text'] = combined_data['Clean_Text'].apply(nfx.remove_stopwords)

x_features = combined_data['Clean_Text']
y_labels = combined_data['emotion']
x_train, x_test, y_train, y_test = train_test_split(x_features, y_labels, test_size=0.3, random_state=42)

w2v_model = Word2Vec.load("path_to_word2vec_model")

# Tokenization and Padding
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(x_train)
x_train_sequences = tokenizer.texts_to_sequences(x_train)
x_test_sequences = tokenizer.texts_to_sequences(x_test)
max_sequence_length = max(len(sequence) for sequence in x_train_sequences)
x_train_padded = pad_sequences(x_train_sequences, maxlen=max_sequence_length, padding='post')
x_test_padded = pad_sequences(x_test_sequences, maxlen=max_sequence_length, padding='post')

# Label Encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)
y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)


# Build the LSTM model
adam = Adam(learning_rate=0.005)

model = Sequential()
model.add(Embedding(vocabSize, 200, input_length=x_train_sequences.shape[1], weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2)))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train_padded, y_train_categorical, epochs=10, batch_size=32, validation_data=(x_test_padded, y_test_categorical))

# Make predictions on the test set
y_pred_categorical = model.predict(x_test_padded)
y_pred_encoded = np.argmax(y_pred_categorical, axis=1)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("\nClassification Report:\n", report)

