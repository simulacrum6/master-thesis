import itertools
import os
import sys
from functools import partial
from multiprocessing import cpu_count, Pool

import librosa
import numpy as np
import pandas as pd
import soundfile

from espnet.transform.spec_augment import freq_mask, time_mask


def spectrogram(y):
    """Converts the given waveform data into a spectrogram."""
    return np.abs(librosa.stft(y, n_fft=1024))


def waveform(spectrogram):
    """Converts the given spectrogram into a waveform."""
    return librosa.griffinlim(spectrogram, n_iter=64)


def extend_file_name(file_name, extension):
    name, ext = os.path.splitext(file_name)
    return f'{name}_{extension}{ext}'


def write_tsv(df, out_dir, file_name, factor, intensity):
    out_file = extend_file_name(f'aug_{file_name}', f'{float(factor)}x_{intensity}')
    out = os.path.join(out_dir, out_file)
    df.to_csv(out, index=False, header=True, sep='\t')


def augment_row(i_row, data_dir, clip_dir, sr, indices, aug_desc, f_mask, t_mask):
    i, row = i_row
    file_name = row['path']
    path = os.path.join(data_dir, clip_dir, file_name)
    spec = spectrogram(librosa.load(path, sr=sr)[0])
    synthetic_samples = [f_mask(t_mask(spec.T)).T for _ in indices]
    entries = []
    for j, sample in enumerate(synthetic_samples):
        suffix = f'{aug_desc}_{j}'
        entry = row.copy()
        entry['id'] = entry['id'] + f'_{suffix}'
        entry['path'] = extend_file_name(file_name, suffix)
        entry['synthetic'] = True
        entries.append(entry)
        out = os.path.join(data_dir, clip_dir, entry['path'])
        soundfile.write(out, waveform(sample), sr, subtype='PCM_24')
    return entries


if __name__ == '__main__':
    # handle args
    data_dir = sys.argv[1]
    dataset_file = sys.argv[2]
    clip_dir = 'clips'
    seperator = '\t'

    # create largest augmentation set
    sr = 16000
    nfft = 1024
    augmentation_factor = 2 ** 3
    augmentation_factor_name = '8.0x'
    augmentations = [freq_mask, time_mask]

    # SpecAugment LD parameters
    T = 100
    F = 27
    n_mask = 2
    value = 'm'  # mean

    # note: number of channels in spectrogram is nfft/2.
    # to achieve same results as SpecAugment, mask sizes are scaled to the same area as in SpecAugement.
    # since spectrograms are not normalized, masks are replaced with mean, not zero.
    f_mask = partial(freq_mask, F=F / 80 * nfft / 2, n_mask=n_mask, replace_with_zero=False)
    t_mask = partial(time_mask, T=T, n_mask=n_mask, replace_with_zero=False)
    augmentation_description = f'TM-{T}-{n_mask}-{value}--FM-{F}-{n_mask}-{value}'
    augmentation_intensity = 'medium'
    indices = range(augmentation_factor)  # used to name output files

    S = pd.read_csv(os.path.join(data_dir, dataset_file), sep=seperator)  # source set

    aug = partial(augment_row,
                  data_dir=data_dir, clip_dir=clip_dir, sr=sr,
                  indices=indices, aug_desc=augmentation_description,
                  f_mask=f_mask, t_mask=t_mask)
    with Pool(cpu_count()) as pool:
        entries = pool.map(aug, S.iterrows())

    entries = list(itertools.chain.from_iterable(entries))  # flatten list of lists
    A = pd.DataFrame(entries).sort_index()
    write_tsv(A, data_dir, dataset_file, augmentation_factor, augmentation_intensity)

    # create subsets
    factors = [
        2 ** 0,
        2 ** 1,
        #    2 ** 2
    ]
    for aug_factor in reversed(factors):
        mask = A['path'].str.match(f'.+_[0-{aug_factor - 1}]\.wav')
        A = A[mask]
        write_tsv(A, data_dir, dataset_file, aug_factor, augmentation_intensity)

    # create smallest subset
    aug_factor = 2 ** -1
    A = A[::2]
    write_tsv(A, data_dir, dataset_file, aug_factor, augmentation_intensity)
