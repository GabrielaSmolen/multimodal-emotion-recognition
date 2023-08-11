import librosa
import librosa.display
import numpy as np
from numpy import trapz


def zero_crossing(x):
    zero_crossings = librosa.zero_crossings(x, pad=False)
    return sum(zero_crossings)


def spectral_centroid(x, sr):
    return librosa.feature.spectral_centroid(y=x, sr=sr)[0]


def spectral_rolloff(x, sr):
    return librosa.feature.spectral_rolloff(y=x, sr=sr)[0]


def mfcc(x, sr):
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
    return mfcc


def mfcc_delta(mfcc):
    delta = librosa.feature.delta(mfcc)
    mean_delta = np.mean(delta, axis=1)
    return mean_delta


def mfcc_delta2(mfcc):
    delta2 = librosa.feature.delta(mfcc, order=2)
    mean_delta2 = np.mean(delta2, axis=1)
    return mean_delta2


def mfcc_max_min(mfcc):
    diffs = abs(np.max(mfcc, axis=1) - np.min(mfcc, axis=1))
    return diffs


def mfcc_std(mfcc):
    std_mfcc = np.std(mfcc, axis=1)
    return std_mfcc


def get_auc(array):
    auc = trapz(array, dx=5)
    return auc


def mean(array):
    return np.mean(array)


def std(array):
    return np.std(array)


def percentile(array, p):
    return np.percentile(array, p)


def max_ptp_value(array):
    max_value = np.max(array)
    min_value = np.min(array)
    value = abs(max_value-min_value)
    return value


def melspectrogram(x, sr):
    spectrogram = librosa.feature.melspectrogram(y=x, sr=sr)
    return spectrogram


if __name__ == "__main__":
    x, sr = librosa.load("data/IEMOCAP_full_release/Session1/dialog/wav/Ses01F_impro01.wav")

    zero_crossing = zero_crossing(x)
    spectral_centroids = spectral_centroid(x, sr)
    spectral_rolloff = spectral_rolloff(x, sr)
    mfcc = mfcc(x, sr)
    mfcc_delta = mfcc_delta(mfcc)
    diff = mfcc_max_min(mfcc)
    std_mfcc = mfcc_std(mfcc)
    spectrogram = melspectrogram(x, sr)
    mean_spect = np.mean(spectrogram[0:40, :], axis=1)
    auc_centroids = get_auc(spectral_centroids)
    auc_rolloff = get_auc(spectral_rolloff)
    mean_centroids = mean(spectral_centroids)
    mean_rolloff = mean(spectral_rolloff)
    std_centroids = std(spectral_centroids)
    std_rolloff = std(spectral_rolloff)
    percentile_25 = percentile(spectral_centroids, 25)