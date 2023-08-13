import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/features.csv")
data = data.loc[data["Label"] != "xxx"]
data = data.loc[data["Label"] != "sur"]
data = data.loc[data["Label"] != "oth"]
data = data.loc[data["Label"] != "fea"]
data = data.loc[data["Label"] != "dis"]

# Determine which classes need augmentation
target_samples_per_class = 2000
class_counts = data['Label'].value_counts()
classes_to_augment = class_counts[class_counts < target_samples_per_class].index

# Augmentation loop
augmented_data = []

for emotion_class in classes_to_augment:
    class_samples = data[data['Label'] == emotion_class]
    num_samples_needed = target_samples_per_class - len(class_samples)

    for _ in range(num_samples_needed):
        # Select a random sample from the class
        sample = class_samples.sample(n=1)

        # Apply random perturbations to features
        augmented_features = sample.drop(["Label", "ID"], axis=1)  # Exclude the label column
        augmented_features += np.random.normal(0, 0.1, size=augmented_features.shape)  # Example perturbation

        # Append the augmented sample to the augmented_data list
        augmented_sample = pd.concat([augmented_features, sample['Label']], axis=1)
        augmented_data.append(augmented_sample)

# Combine original and augmented data
augmented_data = pd.concat([data] + augmented_data)

# Save augmented data back to CSV
augmented_data.to_csv("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/augmented_data.csv", index=False)
