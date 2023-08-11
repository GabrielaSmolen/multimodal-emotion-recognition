from pathlib import Path

import pandas as pd


def extract_labels(file_path: Path) -> pd.DataFrame:
    """
    Extract labels from text files.
    Args:
        file_path (Path): Path to txt files
    Returns:
        extracted labels (pd.DataFrame): Dataframe with extracted labels including start time, end time, wav file name,
        emotion and average dimensional evaluation
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


def main():
    all_labels = pd.DataFrame(columns=["start_time", "end_time", "wav_file", "emotion", "val", "act", "dom"])
    for i in range(1, 6):
        paths = Path(f"C:/Users/Gabi/PycharmProjects/multimodal-emotion-recognition/data/IEMOCAP_full_release/Session{i}/dialog/EmoEvaluation/").glob("*.txt")
        for path in list(paths):
            labels = extract_labels(path)
            all_labels = pd.concat([all_labels, labels], ignore_index=True)
    return all_labels


if __name__ == '__main__':
    main()
