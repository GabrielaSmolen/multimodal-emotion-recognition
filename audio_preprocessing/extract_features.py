from os import listdir
import audio_preprocessing as fe
import librosa
from os.path import isfile, join
import pandas as pd
import numpy as np
import tqdm
from extract_labels import main as extract_labels


def feature_extraction(audio_path):
    x, sr = librosa.load(audio_path)
    features = np.zeros((77,))
    # features[0] = audio_path.split("\\")[-1].split(".")[0]
    features[1] = fe.zero_crossing(x)
    spectral_centroid = fe.spectral_centroid(x, sr)
    spectral_rolloff = fe.spectral_rolloff(x, sr)
    features[2] = fe.get_auc(spectral_centroid)
    features[3] = fe.get_auc(spectral_rolloff)
    features[4] = fe.mean(spectral_centroid)
    features[5] = fe.mean(spectral_rolloff)
    features[6] = fe.std(spectral_centroid)
    features[7] = fe.std(spectral_rolloff)
    features[8] = fe.max_ptp_value(spectral_centroid)
    features[9] = fe.max_ptp_value(spectral_rolloff)
    signal_mfcc = fe.mfcc(x, sr)
    features[10] = fe.percentile(spectral_centroid, 25)
    features[11] = fe.percentile(spectral_rolloff, 25)
    features[12:25] = np.mean(signal_mfcc, axis=1)
    features[25:38] = fe.mfcc_delta(signal_mfcc)
    features[38:51] = fe.mfcc_delta2(signal_mfcc)

    features[51:64] = fe.mfcc_max_min(signal_mfcc[:, 6:-6])
    features[64:77] = fe.mfcc_std(signal_mfcc[:, 6:-6])
    return features


if __name__ == "__main__":
    labels = extract_labels()
    wav_files_path = "C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/audio_preprocessing/new_wav_files"

    means = [f"MFCC mean {x}" for x in range(1, 14)]
    deltas = ["MFCC delta " + str(x) for x in range(1, 14)]
    deltas2 = ["MFCC delta 2 " + str(x) for x in range(1, 14)]
    max_min = ["MFCC max_min " + str(x) for x in range(1, 14)]
    std = ["MFCC std " + str(x) for x in range(1, 14)]

    files = [join(wav_files_path, file) for file in listdir(wav_files_path) if isfile(join(wav_files_path, file))]
    columns = ["ID", "Zero crossings", "Centroid AUC", "Rolloff AUC", "Centroid mean", "Rolloff mean", "Centroid STD",
               "Rolloff STD", "Centroid p-t-p value", "Rolloff p-t-p value", "Centroid_percentile",
               "Rolloff_percentile"] + means + deltas + deltas2 + max_min + std + ["Label"]
    df = pd.DataFrame(columns=columns)
    for file in tqdm.tqdm(files):
        feats = feature_extraction(file)
        feats = list(feats)
        feats[0] = file.split("\\")[-1].split(".")[0]
        label = labels.loc[labels["wav_file"] == feats[0]]["emotion"]
        feats.append(label.values[0])
        df2 = pd.DataFrame([feats], columns=columns)
        df = pd.concat([df, df2])

    df.to_csv(join("C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition", "features.csv"), index=False)
