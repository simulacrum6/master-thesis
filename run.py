from os import makedirs

from evaluation.datasets import main as dataset_stats
from evaluation.preprocess import main as preprocess
from evaluation.visualisations import main as visualize


def create_directories():
    try:
        makedirs('results')
    except FileExistsError:
        pass

    try:
        makedirs('figures')
    except FileExistsError:
        pass


if __name__ == '__main__':
    create_directories()
    preprocess()
    visualize()
    dataset_stats()
