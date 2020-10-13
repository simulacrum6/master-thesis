import os

import pandas as pd


def human_readable(stats):
    stats = stats[(acc_mask(stats))] \
        .drop(columns=['train_set', 'aggregation', 'measure']) \
        .rename(columns={'value': 'accuracy'})
    stats['error_rate'] = 100 - stats['accuracy']
    return stats


def acc_mask(df):
    return (df['aggregation'] == 'mean') & (df['measure'] == 'accuracy')


def assign_baselines(stats):
    # extract baselines
    aug_baseline_mask = (stats['growth_factor'] == 0) & (stats['run'] == 'final')
    test_baseline_mask = (stats['test_set'] == 'test-clean') & aug_baseline_mask
    gen_baseline_mask = (stats['test_set'] == 'test-clean') & (stats['run'] == 'final')
    test_idx_cols = ['train_size', 'metric']
    aug_idx_cols = test_idx_cols + ['test_set']
    gen_idx_cols = test_idx_cols + ['growth_factor']
    baseline_cols = ['error_rate', 'accuracy']

    aug_baselines = stats[aug_baseline_mask].set_index(aug_idx_cols)[baseline_cols]
    test_baselines = stats[test_baseline_mask].set_index(test_idx_cols)[baseline_cols]
    gen_baselines = stats[gen_baseline_mask].set_index(gen_idx_cols)[baseline_cols]

    stats = stats.set_index(aug_idx_cols).join(aug_baselines, rsuffix='__baseline__aug').reset_index()
    stats = stats.set_index(test_idx_cols).join(test_baselines, rsuffix='__baseline__test').reset_index()
    stats = stats.set_index(gen_idx_cols).join(gen_baselines, rsuffix='__baseline__gen').reset_index()

    return stats


def calculate_deltas(stats):
    stats = stats.copy()
    for ref in ['aug', 'test', 'gen']:
        for metric in ['error_rate', 'accuracy']:
            baseline = f'{metric}__baseline__{ref}'
            val = f'{metric}'
            delta = f'{metric}__delta_abs__{ref}'
            rel = f'{metric}__delta_rel__{ref}'
            stats[delta] = stats[val] - stats[baseline]
            stats[rel] = stats[delta] / stats[baseline]
    return stats


def generate_deltas():
    stats = pd.read_csv('results/raw/stats.tsv', sep='\t')
    stats = human_readable(stats)
    stats.to_csv('results/stats_simple.csv', index=False)
    stats = assign_baselines(stats)
    stats = calculate_deltas(stats)

    # split into cer and wer
    cer = stats['metric'] == 'cer'
    wer = ~cer
    stats[cer].to_csv('results/deltas_cer.csv', index=False)
    stats[wer].to_csv('results/deltas_wer.csv', index=False)

    # compute aggregate statistics
    delta_cols = [f'{measure}__delta_{scale}__{ref}'
                  for measure in ['accuracy', 'error_rate']
                  for ref in ['test', 'aug', 'gen']
                  for scale in ['abs', 'rel']]
    cer_deltas = stats.loc[cer, delta_cols + ['test_set']].groupby('test_set').describe()
    wer_deltas = stats.loc[wer, delta_cols + ['test_set']].groupby('test_set').describe()

    cer_deltas.to_csv('results/deltas_cer_stats.csv')
    wer_deltas.to_csv('results/deltas_wer_stats.csv')

    # split into "have improved" and "have not improved"
    improved = stats['accuracy__delta_abs__test'] > 0
    improved_wer = stats.loc[(wer & improved), delta_cols + ['test_set']].groupby('test_set').describe()
    improved_wer.to_csv('results/deltas_wer_stats__improvements.csv')
    improved_cer = stats.loc[(cer & improved), delta_cols + ['test_set']].groupby('test_set').describe()
    improved_cer.to_csv('results/deltas_cer_stats__improvements.csv')


def calculate_loss_variation():
    lcs = pd.read_csv('results/raw/learning_curves.csv').sort_values(by=['train_size', 'growth_factor', 'Step'])
    train_loss_100_8 = (lcs['train_size'] == 100) & (lcs['growth_factor'] == 8) & (lcs['tag'] == 'main_loss')
    stats_100_8 = (lcs[train_loss_100_8]['Value']).diff().describe().rename('Train Loss (100,8)')
    stats_others = (lcs[~train_loss_100_8]['Value']).diff().describe().rename('Train Loss (100,<8)')
    pd.DataFrame([stats_100_8, stats_others]).to_csv('results/lc_example_stats.csv')


def main():
    try:
        os.makedirs('results')
    except FileExistsError:
        pass
    generate_deltas()
    calculate_loss_variation()


if __name__ == '__main__':
    main()
