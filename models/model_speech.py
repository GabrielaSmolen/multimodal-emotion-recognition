import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import LSTM, Input, Flatten, Bidirectional, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2

import keras
from keras.callbacks import EarlyStopping

df = pd.read_csv("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/augmented_data.csv")

# shuffle the dataset! 
df = df.sample(frac=1).reset_index(drop=True)


Y = df["Label"]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Y_onehot = to_categorical(encoded_Y)

X = df.drop(["Label", "ID"], axis=1)

# Initialize MinMaxScaler
scaler = StandardScaler()

# Scale the features
X_scaled = scaler.fit_transform(X)

# Combine scaled features with "Label" column
combined_data = np.hstack((X_scaled, np.array(df["Label"]).reshape(-1, 1)))

# Convert the combined data back to a DataFrame
cols = list(X.columns) + ["Label"]
df_combined = pd.DataFrame(combined_data, columns=cols)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_onehot, test_size=0.2, random_state=42)

# Create a neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10000
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/models/saved_models/speech/trained_model_1.h5")
print("Model saved successfully!")

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Print the training accuracy
train_accuracy = history.history['accuracy'][-1]
print(f"Train Accuracy: {train_accuracy:.4f}")
