import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import LSTM, Input, Flatten, Bidirectional, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.callbacks import EarlyStopping

df = pd.read_csv("/augmented_data.csv")

# shuffle the dataset! 
df = df.sample(frac=1).reset_index(drop=True)


Y = df["Label"]
X = df.drop(["Label", "ID"], axis=1)

print(X.shape)
print(Y.shape)

# convert to numpy arrays
X = np.array(X)

Y.head()

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Y_onehot = to_categorical(encoded_Y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_onehot, test_size=0.2, random_state=42)

# Create a neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(50,)),
    Dropout(0.5),  # Adding dropout for regularization
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(6, activation='softmax')  # Output layer with softmax activation for multi-class classification
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

print()
