import os
import re
import sys
from collections import namedtuple

import pandas as pd

ResultDesc = namedtuple('ResultDescription', ['file', 'metric'])
fields = [
    'speaker',
    'num_sentences',
    'num_words',
    'correct',
    'substitution_errors',
    'deletion_errors',
    'insertion_errors',
    'total_errors',
    'something_with_s_errors'
]
Row = namedtuple('Row', fields)

meta_fields = [
    'train_set',
    'train_size',
    'growth_factor',
    'run',
    'test_set'
]
MetaData = namedtuple('DatasetMetaData', meta_fields)

PATH_START = '|exp/'
SPEAKER_START = '|SPKR'
AVG_START = '|Sum/Avg'

SUBSET_MARKER = 'hrs'
AUG_MARKER = '_medium'


def extract_info_from_desc(desc):
    train_set, tail = desc.split('_', 1)
    info, config, vocab, run = tail.rsplit('_', 3)
    return train_set, info, run


def extract_conditions(info, default_growth_factor=0.0, default_train_size=100):
    if info.endswith(AUG_MARKER):
        info, _ = info.rsplit(AUG_MARKER, 1)
        if info.endswith('aug_only'):
            growth_factor = -1.0
            info = info.replace('_aug_only', '')
        else:
            info, auginfo = info.rsplit('_', 1)
            growth_factor = float(auginfo[:-1])
    else:
        growth_factor = default_growth_factor

    if info.endswith(SUBSET_MARKER):
        info, subsetinfo = info.rsplit('_', 1)
        train_size = float(subsetinfo.replace(SUBSET_MARKER, ''))
    else:
        train_size = default_train_size

    return train_size, growth_factor


def extract_meta_data(path):
    parts = path.split('/')
    desc = parts[2]
    test_set = parts[3].split('_')[1]
    train_set, info, run = extract_info_from_desc(desc)
    train_size, growth_factor = extract_conditions(info)
    return train_set, train_size, growth_factor, run, test_set


def extract_info_from_lc_name(name):
    desc, value_name = name.split('-tag-')
    value_name = os.path.splitext(value_name)[0]
    train_set, info, run = extract_info_from_desc(desc[4:])
    train_size, growth_factor = extract_conditions(info)
    return train_set, train_size, growth_factor, run, value_name


def make_lc_df(file, curve_dir, info_keys):
    info_vals = extract_info_from_lc_name(file)
    df = pd.read_csv(os.path.join(curve_dir, file))
    for key, value in zip(info_keys, info_vals):
        df[key] = value
    return df


def convert_learning_curves():
    curve_dir = 'results/libri/raw/learning_curves'
    if os.path.exists(curve_dir):
        files = os.listdir(curve_dir)
        info_keys = ['train_set', 'train_size', 'growth_factor', 'run', 'tag']
        dfs = [make_lc_df(file, curve_dir, info_keys) for file in files]
        pd.concat(dfs).to_csv('results/libri/learning_curves.csv', index=False)
    else:
        print(f'Learning curve directory does not exist. Make sure it is present at {curve_dir}')


if __name__ == '__main__':
    result_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else result_dir

    descs = [
        ResultDesc('result.txt', 'CER'),
        ResultDesc('result.wrd.txt', 'WER')
    ]

    speaker_rows = []
    for desc in descs:
        with open(os.path.join(result_dir, desc.file), 'r') as f:
            line = f.readline()

            # extract meta info
            while not line.replace(' ', '').startswith(PATH_START):
                line = f.readline()
            meta_data = extract_meta_data(line)

            while not line.replace(' ', '').startswith(SPEAKER_START):
                line = f.readline()
            f.readline()
            line = f.readline()
            while not line.replace(' ', '').startswith(AVG_START):
                f.readline()
                vals = [x.strip() for x in re.split('[ |]+', line)[1:-1]]
                vals.append(desc.metric)
                vals.extend(meta_data)
                speaker_rows.append(vals)
                line = f.readline()

    fields.append('metric')
    fields.extend(meta_fields)
    speaker_stats = pd.DataFrame(speaker_rows, columns=fields).set_index(fields[0])
    meta = '_'.join(str(x) for x in meta_data)
    filename = f'results__{meta}.tsv'
    speaker_stats.to_csv(os.path.join(out_dir, filename), sep='\t')
    speaker_stats = pd.read_csv(os.path.join(out_dir, filename), sep='\t')

    cer = speaker_stats['metric'] == 'CER'
    wer = ~cer

    total_cer = speaker_stats[cer].describe()
    total_cer.index.rename('measure', True)
    total_wer = speaker_stats[wer].describe()
    total_wer.index.rename('measure', True)

    for key, value in zip(meta_fields, meta_data):
        total_cer[key] = value
        total_wer[key] = value

    total_cer.to_csv(os.path.join(out_dir, f'cer_stats__{meta}.tsv'), sep='\t')
    total_wer.to_csv(os.path.join(out_dir, f'wer_stats__{meta}.tsv'), sep='\t')

    convert_learning_curves()
