import os
import sys

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # handle args
    data_dir = sys.argv[1]
    name_prefix = sys.argv[2]
    feature_dir = sys.argv[3]
    seperator = '\t'
    subset_suffix = 'hrs.tsv'
    augset_suffix = 'x_medium.tsv'

    files = [f for f in os.listdir(data_dir)]
    subsets = [f for f in files if f.endswith(subset_suffix)]
    augsets = [f for f in files if f.endswith(augset_suffix)]
    subsets.append(f'{name_prefix}.tsv')

    for subset_name in subsets:
        for augset_name in augsets:
            # generate name
            sub_suffix = subset_name.replace(name_prefix, '').replace('.tsv', '')
            aug_suffix = augset_name.replace(f'aug_{name_prefix}', '').replace('.tsv', '')
            combined_name = f'{name_prefix}{sub_suffix}{aug_suffix}'

            # create directory
            out_dir = os.path.join(feature_dir, combined_name)
            try:
                os.makedirs(out_dir)
            except FileExistsError:
                pass

            # read data
            subset = pd.read_csv(os.path.join(data_dir, subset_name), sep=seperator)
            augset = pd.read_csv(os.path.join(data_dir, augset_name), sep=seperator)

            # find appropriate subset
            subset_ids = subset[['id']]
            mask = np.isin(augset['id'].str.split('_', 1).str.get(0).values, subset['id'].values)
            augset_ids = augset.loc[mask, ['id']]

            # add start of line filter for grep
            subset_ids['id'] = '^' + subset_ids['id']
            augset_ids['id'] = '^' + augset_ids['id']

            # store ids
            subset_ids.to_csv(os.path.join(out_dir, 'regular.txt'), index=False, header=False)
            augset_ids.to_csv(os.path.join(out_dir, 'augmented.txt'), index=False, header=False)

            # create augmentation only and no augmentation
            if aug_suffix.startswith('_1.0x'):
                # aug only
                intensity = aug_suffix[1:].split('_')[1]
                out_dir = os.path.join(feature_dir, f'{name_prefix}{sub_suffix}_aug_only_{intensity}')
                try:
                    os.makedirs(out_dir)
                except FileExistsError:
                    pass
                open(os.path.join(out_dir, 'regular.txt'), 'w').close()  # create empty file
                augset_ids.to_csv(os.path.join(out_dir, 'augmented.txt'), index=False, header=False)

                # regular only
                out_dir = os.path.join(feature_dir, f'{name_prefix}{sub_suffix}')
                try:
                    os.makedirs(out_dir)
                except FileExistsError:
                    pass
                subset_ids.to_csv(os.path.join(out_dir, 'regular.txt'), index=False, header=False)
                open(os.path.join(out_dir, 'augmented.txt'), 'w').close()  # create empty file
