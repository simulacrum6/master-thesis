import os
import sys
from multiprocessing import cpu_count, Pool

import librosa
import pandas as pd


def get_duration(file_path, sr=16000):
    waveform, _ = librosa.load(file_path, sr=sr)
    duration = librosa.get_duration(waveform, sr)
    return duration


def extend_filename(filename):
    name, ext = os.path.splitext(filename)
    return f'{name}_ext{ext}'


if __name__ == '__main__':
    # handle args
    data_dir = sys.argv[1]
    dataset_files = sys.argv[2:] if len(sys.argv) > 2 else [f for f in os.listdir(data_dir) if f.endswith('.tsv')]
    clip_dir = 'clips'
    seperator = '\t'

    for dataset_file in dataset_files:
        df = pd.read_csv(os.path.join(data_dir, dataset_file), sep=seperator)

        # add id and synthetic identifier
        if os.path.basename(data_dir) == 'libri':
            df['id'] = df['path'].str.replace('.wav', '')
        else:
            df['id'] = df['client_id'].astype(str) + '-' + df['path']
            df['id'] = df['id'].str.split('.').str.get(0)
        df['synthetic'] = False

        # extract durations
        df = df.astype(dtype={'id': 'category', 'client_id': 'category', 'path': str, 'sentence': str})
        files = [os.path.join(data_dir, clip_dir, name) for name in df['path']]
        with Pool(cpu_count()) as pool:
            dur = pool.map(get_duration, files)

        # store extended dataset
        durs = pd.Series(dur, name='duration')
        df['duration'] = durs
        out = os.path.join(data_dir, extend_filename(dataset_file))
        df.to_csv(out, header=True, index=False, sep=seperator)
