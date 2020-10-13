import os
import sys

import pandas as pd

if __name__ == '__main__':
    # handle args
    data_dir = sys.argv[1]
    dataset_file = sys.argv[2]
    clip_dir = 'clips'
    seperator = '\t'

    df = pd.read_csv(os.path.join(data_dir, dataset_file), sep=seperator)

    # subset sizes in seconds. 50 hrs, 10 hrs, 5 hrs, 1 hrs, 0.5 hrs, 0.1 hrs
    sizes = [
        #  ('50hrs', 3.6e5 / 2),
        ('10hrs', 3.6e4),
        ('5hrs', 3.6e4 / 2),
        ('1hrs', 3.6e3),
        ('0.1hrs', 3.6e2)
    ]

    current_set = df

    # select subsets
    for duration_name, duration_target in sizes:
        shuffled = current_set.sample(frac=1)
        duration_total = 0
        entries = []
        for i, row in shuffled.iterrows():
            entries.append(row)
            duration_total += row['duration']
            if duration_total >= duration_target:
                break

        subset = pd.DataFrame(entries)
        extended_name = f'{os.path.splitext(dataset_file)[0]}_{duration_name}.tsv'
        out = os.path.join(data_dir, extended_name)
        subset.sort_index().to_csv(out, index=False, header=True, sep=seperator)
        current_set = subset
