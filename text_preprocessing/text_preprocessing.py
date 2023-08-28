import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from os.path import join
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path

from extract_labels import main as extract_labels


def process_text(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in cleaned_tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return lemmatized_tokens


if __name__ == '__main__':
    # Download NLTK resources (only required once)
    # nltk.download("punkt")
    # nltk.download("stopwords")
    # nltk.download("wordnet")
    labels = extract_labels()
    columns = list(labels.columns) + ["sentence"] + ["words"]
    result = pd.DataFrame(columns=columns)
    for i in range(1, 6):
        text_files_paths = Path(f"C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/data/IEMOCAP_full_release/Session{i}/dialog/transcriptions/").glob("*.txt")
        for path in list(text_files_paths):
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    line_id = line.split("[")[0].strip()
                    info = labels.loc[labels["wav_file"] == line_id]
                    # result = pd.concat([result, info])
                    text_line = line.split(":")[-1]
                    words = process_text(text_line)
                    info["sentence"] = text_line
                    info["words"] = [words]
                    result = pd.concat([result, info])

    emotions_to_exclude = ["xxx", "sur", "oth", "fea", "dis"]

    result = result.loc[~result["emotion"].isin(emotions_to_exclude)]
    result = result.dropna(subset=["emotion"])
    result = result.drop(["start_time", "end_time", "val", "act", "dom", "words"], axis=1)

    result.to_csv(join("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition", "text_data.csv"), index=False)
