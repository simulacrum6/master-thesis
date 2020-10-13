# Investigating Effects of Data Quantityand Augmentation Methods for Speech-To-Text Systems

This repository contains the code to run experiments and evaluations described in the thesis _Investigating Effects of Data Quantityand Augmentation Methods for Speech-To-Text_.


## Prerequisites

Experiments depend on [ESPnet](https://github.com/espnet/espnet).
To run the experiments, follow the [installation instructions](https://espnet.github.io/espnet/installation.html).
I recommend using the [docker container](https://espnet.github.io/espnet/docker.html) provided by ESPnet.

After installing ESPnet, copy files from the `asr1` folder to the `commonvoice` asr recipe folder.

```bash
cp -r asr1/. <espnet-dir>/egs/commonvoice/asr1/
```

## Running the Experiments

To run experiments, execute the `run.sh`.

```bash
cd <espnet-dir>/egs/commonvoice/asr1/
./run.sh > log 2> err
```

**Note:** You may need to activate the conda environment provided with the docker container, before starting the experiments.

```bash
conda ./<espnet-dir>/tools/venv/bin/activate
```

## Results

Results will be collected in `<asr1>/tensorboard` and `<asr1>/results`.

Results of the experiments are also included in this repository.
In case you modify experiments and generate the same outputs, you need to place results into the repository directory under `results/raw/` and run `run.py`.

To generate the outputs the [Seaborn](https://seaborn.pydata.org/) and [Librosa](https://librosa.org/) packages are required.
