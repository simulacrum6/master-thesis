import os
import shutil
import sys

import pandas as pd


def subdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


if __name__ == '__main__':
    # handle args
    nargs = len(sys.argv) - 1
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    split_dirs = sys.argv[3:]

    clip_dir = os.path.join(out_dir, 'clips')
    speaker_file = 'SPEAKERS.TXT'
    extension = '.tsv'
    separator = '\t'
    audio_format = '.flac'
    gender_map = {'F': 'female', 'M': 'male'}
    NA = '-'

    speakers = pd.read_csv(os.path.join(data_dir, speaker_file), sep='|',
                           comment=';', names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'], )
    speakers['ID'] = speakers['ID'].astype(int)
    for col in ['SEX', 'SUBSET', 'NAME']:
        speakers[col] = speakers[col].str.strip()

    try:
        os.makedirs(clip_dir)
    except FileExistsError:
        pass

    # create tsv files and copy source files for every clip.
    for split in split_dirs:
        entries = []
        for speaker in subdirs(os.path.join(data_dir, split)):
            for chapter in subdirs(os.path.join(data_dir, split, speaker)):
                cwd = os.path.join(data_dir, split, speaker, chapter)
                transcript_file = f'{speaker}-{chapter}.trans.txt'
                gender = speakers.query(f'ID == {speaker} & SUBSET == "{split}"')['SEX'].values[0]
                with open(os.path.join(cwd, transcript_file), 'r') as f:
                    for line in f:
                        id, transcript = line.split(' ', 1)
                        filename = f'{id}{audio_format}'
                        entry = {
                            'client_id': speaker,
                            'path': filename,
                            'sentence': transcript.replace('\n', ''),
                            'up_votes': NA,
                            'down_votes': NA,
                            'age': NA,
                            'gender': gender_map.get(gender, '-'),
                            'accent': NA
                        }
                        entries.append(entry)
                        # copy files
                        shutil.copy(os.path.join(cwd, filename), clip_dir)
        pd.DataFrame(entries).to_csv(os.path.join(out_dir, f'{split}{extension}'), index=False, sep=separator)
