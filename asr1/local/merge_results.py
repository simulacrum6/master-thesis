import os
import sys

import pandas as pd


def concat_files(result_dir, prefix):
    files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.startswith(prefix)]
    return pd.concat([pd.read_csv(f, sep='\t') for f in files])


if __name__ == '__main__':
    result_dir = sys.argv[1]
    out_dir = sys.argv[2]
    prefixes = ['results', 'wer_stats', 'cer_stats']

    prefix = 'results'
    results = concat_files(result_dir, prefix)
    out = os.path.join(out_dir, f'{prefix}.tsv')
    results.to_csv(out, sep='\t', index=False, header=True)

    prefix = 'wer_stats'
    wer_stats = concat_files(result_dir, prefix)
    wer_stats['metric'] = 'wer'

    prefix = 'cer_stats'
    cer_stats = concat_files(result_dir, prefix)
    cer_stats['metric'] = 'cer'

    # merge cer and wer, rename columns and convert to proper long-form format
    prefix = 'stats'
    out = os.path.join(out_dir, f'{prefix}.tsv')
    col_mapping = {'correct': 'accuracy', 'something_with_s_errors': 'total_errors', 'measure': 'aggregation'}
    keep_cols = ['metric', 'aggregation', 'train_set', 'train_size', 'growth_factor', 'test_set', 'run']
    pd.concat([wer_stats, cer_stats]) \
        .drop(columns=['speaker']) \
        .rename(columns=col_mapping) \
        .melt(id_vars=keep_cols, var_name='measure') \
        .to_csv(out, sep='\t', index=False)
