import os
import sys
from multiprocessing import cpu_count, Pool

import librosa
import pandas as pd
import soundfile


def convert_to_wav(path, sr=16000):
    y, _ = librosa.load(path, sr)
    soundfile.write(path.replace('.m4a', '.wav'), y, sr, subtype='PCM_24')


if __name__ == '__main__':
    # handle args
    data_dir = sys.argv[1]
    dataset_files = sys.argv[2:] if len(sys.argv) > 2 else [f for f in os.listdir(data_dir) if f.endswith('.tsv')]
    clip_dir = 'clips'
    seperator = '\t'

    for dataset_file in dataset_files:
        df = pd.read_csv(os.path.join(data_dir, dataset_file), sep=seperator)

        # write files
        files = [os.path.join(data_dir, clip_dir, name) for name in df['path']]
        process_count = cpu_count() - 1 if cpu_count() > 1 else 1
        with Pool(process_count) as pool:
            pool.map(convert_to_wav, files)

        # update path in dataset
        df['path'] = df['path'].str.replace('.m4a', '.wav')
        out = os.path.join(data_dir, dataset_file)
        df.to_csv(out, header=True, index=False, sep=seperator)
