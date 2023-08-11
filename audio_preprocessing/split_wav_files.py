import os
from pathlib import Path
from pydub import AudioSegment

from extract_labels import main as extract_labels


def split_wav_files(output_path):
    """
    Extracts audio segments based on specified time intervals and labels.

    This function extracts audio segments based on specified start and end times for each labeled emotion instance,
    and exports the extracted segments to new WAV files in a designated directory.
    """
    for i in range(1, 6):
        paths = Path(f"data/IEMOCAP_full_release/Session{i}/dialog/wav/").glob("*.wav")
        for path in list(paths):
            labels = extract_labels()
            path = str(path)
            for index, row in labels.iterrows():
                if path.split("\\")[-1].split(".")[0] in row["wav_file"]:
                    t1 = float(row["start_time"]) * 1000
                    t2 = float(row["end_time"]) * 1000
                    new_audio = AudioSegment.from_wav(path)
                    new_audio = new_audio[t1:t2]
                    new_audio.export(f"{output_path}{row['wav_file']}.wav", format="wav")


if __name__ == "__main__":
    output_directory = "new_wav_files/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    split_wav_files(output_directory)
