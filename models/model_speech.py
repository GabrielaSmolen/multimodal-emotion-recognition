import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2


df = pd.read_csv("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/augmented_data.csv")

df = df.sample(frac=1).reset_index(drop=True)


Y = df["Label"]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Y_onehot = to_categorical(encoded_Y)

X = df.drop(["Label", "ID"], axis=1)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

combined_data = np.hstack((X_scaled, np.array(df["Label"]).reshape(-1, 1)))

cols = list(X.columns) + ["Label"]
df_combined = pd.DataFrame(combined_data, columns=cols)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_onehot, test_size=0.2, random_state=42)

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

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 1000
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

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

train_accuracy = history.history['accuracy'][-1]
print(f"Train Accuracy: {train_accuracy:.4f}")

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
