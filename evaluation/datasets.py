import itertools
import string
from collections import Counter

import numpy as np
import pandas as pd


def flatten(xs):
    return list(itertools.chain.from_iterable(xs))


def calculate_dataset_stats(df, name):
    sents = df['sentence']
    n_sents = len(sents)
    n_sents_unique = len(sents.unique())
    n_speakers = len(df['client_id'].unique())

    by_speaker = df.groupby('client_id')

    sents_per_speaker = by_speaker.count()['sentence']
    sents_per_speaker_stats = sents_per_speaker.describe()
    sents_per_speaker_mean = sents_per_speaker_stats['mean']

    wordlists = [s.split(' ') for s in sents]
    words = flatten(wordlists)
    n_words = len(words)
    words_unique = np.unique(words)
    n_words_unique = len(words_unique)

    vocab = Counter(words)
    ranking = Counter({pair[0]: i + 1 for i, pair in enumerate(vocab.most_common())})
    for word in vocab:
        vocab[word] /= n_words

    duration_stats = df['duration'].describe()
    duration_mean = duration_stats['mean']
    duration_total = df['duration'].sum()

    duration_per_speaker = by_speaker.sum()['duration']
    duration_per_speaker_mean = duration_per_speaker.mean()

    males = sum(df['gender'] == 'male')
    females = sum(df['gender'] == 'female')
    if females == 0:
        m_f_ratio = -1
    else:
        m_f_ratio = males / females

    df_stats = {
        'Dataset': name,
        'Size (h)': duration_total / 3600,
        'Utterances': n_sents,
        'Unique Utts.': n_sents_unique,
        'Avg. Duration (s)': duration_mean,
        'Tokens': n_words,
        'Types': n_words_unique,
        'TTR': n_words_unique / n_words,
        'Speakers': n_speakers,
        'Utts. / Spkrs.': n_sents / n_speakers,
        'm/f Ratio': m_f_ratio,
    }

    df_per_speaker_stats = {
        'Dataset': name,
        'Size (hrs)': duration_per_speaker_mean / 3600,
        'Avg. Duration (s)': by_speaker.mean().mean()['duration'],
        'Sentences': sents_per_speaker_mean
    }

    return (df_stats, df_per_speaker_stats, (vocab, ranking))


def cleanup():
    df = pd.read_csv('data/dfs/test_ood.tsv', sep='\t')
    allowed = f'{string.ascii_uppercase}\' '
    clean = df['sentence'].str.upper().map(lambda x: ''.join([c for c in x if c in allowed]))
    df['sentence'] = clean
    df.to_csv('data/dfs/test_ood_ext.tsv', sep='\t', index=False)

def main():
    import os
    try:
        os.makedirs('figures')
    except FileExistsError:
        pass

    try:
        os.makedirs('tables')
    except FileExistsError:
        pass

    cleanup()
    data_dir = 'data/dfs'
    files = ['train-clean-100_ext.tsv', 'dev-clean_ext.tsv', 'test-clean_ext.tsv', 'test_ood_ext.tsv']
    datasets = [pd.read_csv(os.path.join(data_dir, f), sep='\t') for f in files]
    TRAIN = 'LS train-clean-100'
    DEV = 'LS dev-clean'
    TEST = 'LS test-clean'
    OOD = 'CommonVoice'
    names = [TRAIN, DEV, TEST, OOD]

    stats = []
    stats_per_speaker = []
    vocab_rankings = []
    for name, dataset in zip(names, datasets):
        stts, stts_pr_sprk, vr = calculate_dataset_stats(dataset, name)
        stats.append(stts)
        stats_per_speaker.append(stts_pr_sprk)
        vocab_rankings.append(vr)

    pd.DataFrame(stats).transpose().to_csv('results/ds_stats.csv', float_format='%.2f')
    pd.DataFrame(stats_per_speaker).transpose().to_csv('results/ds_spkr_stats.csv', float_format='%.2f')

    # some example checks for ranks of the predictions
    vocab = {name: v[0] for name, v in zip(names, vocab_rankings)}
    ranking = {name: v[1] for name, v in zip(names, vocab_rankings)}
    k = 10
    vocab[TRAIN].most_common(k)
    vocab[TEST].most_common(k)
    vocab[OOD].most_common(k)

    coll_10_2 = ['POOR', 'MAN', 'AND', 'SAID', 'DIRECTED', 'THE', 'RAGE', 'AND']
    coll_10_2_ranks = [ranking[TRAIN][w] for w in coll_10_2]
    coll_10_8 = ['SEEN', 'SHE', 'THAT', 'HERE']
    coll_10_8_ranks = [ranking[TRAIN][w] for w in coll_10_8]
    coll_100_8 = ['THEY', 'WENT', 'ON', 'BEEN', 'ONE', 'ELSE', 'HOUSE']
    coll_100_8_ranks = [ranking[TRAIN][w] for w in coll_100_8]

if __name__ == '__main__':
    main()
