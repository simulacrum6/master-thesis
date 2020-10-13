import os
from math import log

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_palette('deep')

gf_order = [-1.0, 0.0, 0.5, 1.0, 2.0, 8.0]

# growth factor to human readable label
gf2lbl = {
    '-1.0': 'Full Aug.',
    '0.0': 'Baseline',
    '0.5': '+1/2x',
    '1.0': '+1x',
    '2.0': '+2x',
    '8.0': '+8x'
}

gf2color = {
    '-1.0': 'blue',
    '0.0': 'orange',
    '0.5': 'green',
    '1.0': 'red',
    '2.0': 'purple',
    '8.0': 'pink'
}

# test set names to human readable label
test2lbl = {
    'test-clean': 'LibriSpeech test-clean (in-domain)',
    'dev-clean': 'LibriSpeech dev-clean',
    'ood': 'CommonVoice (out-of-domain)'
}

# maps metric to unit
metric2lbl = {
    'wer': 'Word',
    'cer': 'Character'
}

run2lbl = {
    'final': '',
    'extended': 'extended'
}

err2lbl = {
    'deletion_errors': 'Deletion Errors',
    'insertion_errors': 'Insertion Errors',
    'substitution_errors': 'Substitution Errors'
}

def hz_to_mel(f):
    return 2595 * log(1 + f / 700, 10)


def label_growth_factor_legend(figure):
    figure.legend.set_title('Growth Factor')
    for t in figure.legend.texts:
        t.set_text(gf2lbl[t._text])


def meta_learning_curves(stats, measure, kind):
    # generate figure
    mask = (stats['aggregation'] == 'mean') & (stats['measure'] == measure) & (stats['run'] == 'final') & (
                stats['test_set'] != 'dev-clean')
    colors = [gf2color[str(gf)] for gf in stats['growth_factor'].unique()]
    kwargs = {'scale': 0.5} if kind == 'point' else {}
    grid = sns.catplot(data=stats[mask], kind=kind, y='value', x='train_size',
                       hue='growth_factor', col='test_set', row='metric', row_order=['wer', 'cer'],
                       col_order=['test-clean', 'ood'], palette='bright', **kwargs)

    # customize axis labels
    grid.set(ylim=(0, 100))
    grid.set_axis_labels(x_var='Training Set Size (h)', y_var=f'Accuracy (%)')

    for ax in grid.axes[0, :]:
        old = ax.get_title()
        test_set = old.rsplit(' = ')[-1]
        ax.set_title(test2lbl[test_set])

    for ax in grid.axes[1, :]:
        old = ax.get_title()
        test_set = old.rsplit(' = ')[-1]
        ax.set_title(test2lbl[test_set])

    for m, ax in zip(['wer', 'cer'], grid.axes[:, 0]):
        unit = metric2lbl[m]
        ax.set(ylabel=f'{unit} Accuracy (%)')

    # customize legend
    label_growth_factor_legend(grid)

    return grid


def plot_and_save_meta_learning_curves():
    stats = pd.read_csv(f'results/raw/stats.tsv', sep='\t')
    formats = ['point', 'bar']
    for form in formats:
        figure = meta_learning_curves(stats, 'accuracy', form)
        figure.savefig(f'figures/acc_by_quantity_{form}.png')


def plot_errors():
    stats = pd.read_csv(f'results/raw/stats.tsv', sep='\t')
    errs = ['substitution_errors', 'insertion_errors']  # , 'deletion_errors']
    mask = (stats['aggregation'] == 'mean') & (stats['run'] == 'final') & (stats['measure'].isin(errs))
    plot_mask = (stats['test_set'] == 'test-clean') & (stats['metric'] == 'wer')
    grid = sns.catplot(data=stats[mask & plot_mask], kind='point', scale=0.5, x='train_size', y='value',
                       hue='growth_factor', col='measure')  # , row='metric')

    grid.set_axis_labels(x_var='Training Set Size (h)', y_var=f'Error (%)')

    for ax in grid.axes[0, :]:
        old = ax.get_title()
        err = old.rsplit(' = ')[-1]
        ax.set_title(err2lbl[err])

    label_growth_factor_legend(grid)
    grid.savefig('figures/error_types.png')


def bin_window(df, window_size, column, bin_suffix='_bin'):
    df = df.copy()
    binned = f'{column}{bin_suffix}'
    df[binned] = np.NaN
    for n in np.arange(len(df) // window_size) * window_size:
        df.loc[n:n + window_size, binned] = df.loc[n:n + window_size, column].mean()
    return df


def plot_and_save_example_learning_curves(curves):
    curves = curves.copy()
    curves = bin_window(curves, 10, 'Step')
    only_example = (curves['train_size'] == 100.0) & (curves['tag'] == 'main_loss')
    colors = [gf2color[str(gf)] for gf in curves['growth_factor'].unique()]
    figure = sns.relplot(data=curves[only_example], x='Step_bin', y='Value', hue='growth_factor', kind='line',
                         palette='bright', aspect=1.6)
    figure.set_axis_labels(x_var='Optimization Step', y_var='Loss')
    label_growth_factor_legend(figure)
    figure.savefig('figures/lc_example.png')


def ql2lbl(quantity_level):
    if quantity_level >= 1:
        quantity_level = int(quantity_level)
    return f'{quantity_level} hrs.'


def delta_str(x):
    return f'{x:+.1f}'


def perc_str(x):
    return f'{x}\\%'


def quantity_experiment_tables():
    cols = [
        'run',
        'train_size',
        'growth_factor',
        'error_rate',
        'error_rate__delta_abs__aug',
        'error_rate__delta_rel__aug'
    ]
    colmap = {
        'run': 'Run',
        'train_size': 'q',
        'growth_factor': 'g',
        'error_rate': 'WER',
        'error_rate__delta_abs__aug': '$\\Delta$',
        'error_rate__delta_rel__aug': '\\%'
    }
    index_cols = ['run', 'train_size', 'growth_factor']
    for metric in ['wer', 'cer']:
        deltas = pd.read_csv(f'results/deltas_{metric}.csv')
        metric_mask = (deltas['metric'] == metric)
        for test_set in ['test-clean']:
            test_set_mask = (deltas['test_set'] == test_set) & (deltas['growth_factor'] >= 0)
            table = deltas[metric_mask & test_set_mask][cols] \
                .sort_values(by=index_cols, ascending=[False, True, True]) \
                .rename(columns=colmap)
            table['q'] = table['q'].map(ql2lbl)
            table['g'] = table['g'].map(lambda x: gf2lbl[str(x)])
            table['Run'] = table['Run'].map(run2lbl)
            table['$\\Delta$'] = table['$\\Delta$'].map(delta_str)
            table['\\%'] = (table['\\%'] * 100).map(delta_str).map(perc_str)
            table.to_csv(f'results/table_qty_{metric}.csv', float_format='%.1f', index=False)


def synthetic_experiment_tables():
    cols = [
        'train_size',
        'error_rate__baseline__aug',
        'error_rate',
        'error_rate__delta_abs__aug',
        'error_rate__delta_rel__aug'
    ]
    colmap = {
        'run': 'Run',
        'train_size': 'q',
        'growth_factor': 'g',
        'error_rate__baseline__aug': 'WER_{\\text{Base}}',
        'error_rate': 'WER_{\\text{F.Aug}}',
        'error_rate__delta_abs__aug': '$\\Delta$',
        'error_rate__delta_rel__aug': '\\%'
    }
    index_cols = ['train_size']
    for metric in ['wer', 'cer']:
        deltas = pd.read_csv(f'results/deltas_{metric}.csv')
        metric_mask = (deltas['metric'] == metric)
        full_aug_mask = (deltas['growth_factor'] == -1.0)
        for test_set in ['test-clean']:
            test_set_mask = (deltas['test_set'] == test_set)
            table = deltas[metric_mask & test_set_mask & full_aug_mask][cols] \
                .sort_values(by=index_cols, ascending=True) \
                .rename(columns=colmap)
            table['q'] = table['q'].map(ql2lbl)
            table['$\\Delta$'] = table['$\\Delta$'].map(delta_str)
            table['\\%'] = (table['\\%'] * 100).map(delta_str).map(perc_str)
            table.to_csv(f'results/table_syn_{metric}.csv', float_format='%.1f', index=False)


def generalisation_experiment_tables():
    cols = [
        'run',
        'train_size',
        'growth_factor',
        'error_rate',
        'error_rate__delta_abs__aug',
        'error_rate__delta_rel__aug'
    ]
    colmap = {
        'run': 'Run',
        'train_size': 'q',
        'growth_factor': 'g',
        'error_rate__baseline__aug': 'WER_{\\text{id}',
        'error_rate': 'WER_{\\text{ood}}',
        'error_rate__delta_abs__aug': '$\\Delta$',
        'error_rate__delta_rel__aug': '\\%'
    }
    index_cols = ['run', 'train_size', 'growth_factor']
    for metric in ['wer', 'cer']:
        deltas = pd.read_csv(f'results/deltas_{metric}.csv')
        metric_mask = (deltas['metric'] == metric)
        for test_set in ['ood']:
            test_set_mask = (deltas['test_set'] == test_set)
            table = deltas[metric_mask & test_set_mask][cols] \
                .sort_values(by=index_cols, ascending=[False, True, True]) \
                .rename(columns=colmap)
            table['q'] = table['q'].map(ql2lbl)
            table['g'] = table['g'].map(lambda x: gf2lbl[str(x)])
            table['Run'] = table['Run'].map(run2lbl)
            table['$\\Delta$'] = table['$\\Delta$'].map(delta_str)
            table['\\%'] = (table['\\%'] * 100).map(delta_str).map(perc_str)
            table.to_csv(f'results/table_gen_results_{metric}.csv', float_format='%.1f', index=False)


def generalisation_experiment_plots():
    stats = pd.read_csv('results/raw/gen_benefit.csv')
    grid = sns.catplot(data=stats, kind='bar', y='Generalisation Score', x='Train Set Size (h)',
                       hue='Growth Factor', palette='bright')
    grid.savefig('figures/gen_scores.png')


def audio_files_demo():
    sns.set_style('white', rc={'axes.grid': False})
    dir = 'data/examples'
    entry = pd.read_csv(f'{dir}/example.tsv', sep='\t').iloc[0, :]
    wav, sr = librosa.load(f'{dir}/{entry["path"]}', sr=16000)
    spec = np.abs(librosa.stft(wav, n_fft=512))
    log_spec = librosa.power_to_db(spec, ref=np.min)
    mel_spec = librosa.feature.melspectrogram(S=spec, n_mels=80, n_fft=512)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.min)
    times = librosa.frames_to_time(np.arange(log_spec.shape[1]), sr=16000, n_fft=512)
    ax = plt.plot(np.arange(len(wav)), wav)
    plt.axes().set(xlabel='Sample', ylabel='Amplitude')
    plt.savefig('figures/w.png')
    ax = plt.matshow(spec[::-1], cmap='viridis')
    plt.savefig('figures/s.png')
    ax = plt.matshow(log_mel_spec[::-1], cmap='viridis')
    plt.savefig('figures/lms.png')
    ax = plt.matshow(log_spec[::-1], cmap='viridis')
    plt.savefig('figures/ms.png')
    sns.set_style(rc={'axes.grid': True})


def plot_hz_to_mel():
    freqs = np.arange(40100)
    mels = [hz_to_mel(f) for f in freqs]
    ax = plt.plot(freqs, mels)
    plt.axes().set(xlabel='Hertz', ylabel='Mel', title='Relationship between Hertz and Mel')
    plt.savefig('figures/hz2mel.png')


def main():
    try:
        os.makedirs('figures')
    except FileExistsError:
        pass

    plot_and_save_meta_learning_curves()
    plot_and_save_example_learning_curves(pd.read_csv('results/raw/learning_curves.csv'))
    quantity_experiment_tables()
    synthetic_experiment_tables()
    generalisation_experiment_tables()
    generalisation_experiment_plots()
    audio_files_demo()
    plot_hz_to_mel()
    plot_errors()


if __name__ == '__main__':
    main()
