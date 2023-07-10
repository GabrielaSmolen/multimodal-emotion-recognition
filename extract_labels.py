import pandas as pd
from pathlib import Path


def extract_labels(file_path: Path) -> pd.DataFrame:
    """
    Extract labels from text files.
    :param file_path: Path to txt files
    :return: extracted labels including start time, end time, wav file name, emotion, val, act, dom
    """
    extracted_labels = pd.DataFrame(columns=["start_time", "end_time", "wav_file", "emotion", "val", "act", "dom"])
    with open(file_path) as file:
        for line in file:
            if line.startswith('[') and line.endswith(']\n'):
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                extracted_labels.loc[len(extracted_labels)] = [start_time.strip(), end_time.strip(), wav_file_name,
                                                               emotion, float(val), float(act), float(dom)]
                
    return extracted_labels


if __name__ == '__main__':
    all_labels = pd.DataFrame(columns=["start_time", "end_time", "wav_file", "emotion", "val", "act", "dom"])
    for i in range(1, 6):
        paths = Path(f"data/IEMOCAP_full_release/Session{i}/dialog/EmoEvaluation/").glob("*.txt")
        for path in list(paths):
            labels = extract_labels(path)
            all_labels = pd.concat([all_labels, labels], ignore_index=True)
