# benchmarking-lvms

Official source code repository for the paper ["Benchmarking Generative Latent Variable Models for Speech"](https://arxiv.org/abs/2202.12707).

The paper was published at the [Deep Generative Models for Highly Structured Data workshop](https://deep-gen-struct.github.io) at ICLR 2022.

## Installation

### Install environment using a pre-innstalled system-level CUDA

```bash
conda deactivate
conda env remove -n blvm -y
conda create -y -n blvm python==3.8
conda activate blvm
env REQUIREMENTS_MODE="torch extra" pip install --upgrade --editable .
```

### Install environment using Conda to get CUDA and torch

```bash
conda deactivate
conda env remove -n blvm -y
conda create -y -n blvm python==3.8
conda activate blvm
conda install -y pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch
env REQUIREMENTS_MODE="notorch extra" pip install --upgrade --editable . 
```

### Install requirements separately

```bash
pip install -r requirements-extra.txt
nbstripout --install
```

## Experiments

**Prepare datasets**:
We provide scripts to use for downloading TIMIT and LibriSpeech (and LibriLight).
TIMIT is licensed and must be downloaded before we can run the preparation script.

```bash
python ./scripts/data/prepare_timit.py
python ./scripts/data/prepare_librispeech.py
python ./scripts/data/prepare_librilight.py
```

**Setup data directory**:
When the first experiment is executed, `blvm` will ask you to specify a root data directory in which the repository will store all datasets, checkpoints etc. This directory will be saved into a file `./BLVM.env`. You can also manually create this file:

```md
# BLVM.env
BLVM_DATA_ROOT_DIRECTORY=/absolute/path/of/root/data/directory
```

### Benchmarks

All run specifications are in the file `./experiments/benchmarks.txt`.
They can be run one at a time or scheduled on a single node using the script `./experiments/schedule_experiments.py` where you can specify the number of runs executing in parallel.

```bash
python schedule_experiments --run_file ./experiments/benchmarks.txt --num_parallel 2
```

### Phoneme classification

All run specifications are in the file `./experiments/phonemes.txt`.

**Create short subsets of TIMIT for phoneme classification**:

```bash
python scripts/data/create_short_subsets.py --dataset "timit" --source "train.txt" --lengths 57600000 9600000 --names "train_1h" "train_10m"
```

**Dump representations from WaveNet and LSTM models**:

```bash
python experiments/dump_representations.py --extra s64 --entity blvm --project wavenet --id "model-run-id"
```

## Test

To run tests, execute

```bash
pytest -sv --cov --cov-report=term tests
```

## Data directory structure

```text
root_dir
    data/
        librispeech/
        timit/
    source/
        librispeech/
        timit/
```

## wandb

We track experiments using wandb.

### Enable/Disable

- `wandb online`, `WANDB_MODE=online` or `wandb.init(mode="online")` - runs in online mode, the default
- `wandb offline`, `WANDB_MODE=offline` or `wandb.init(mode="offline")` - runs in offline mode, writes all data to disk for later syncing to a server
- `wandb disabled`, `WANDB_MODE=disabled` or `wandb.init(mode="disabled")` - makes all calls to wandb api's noop's, while maintaining core functionality such as wandb.config and wandb.summary in case you have logic that reads from these dicts.
- `wandb enabled`, `WANDB_MODE=` or `wandb.init(mode="enabled")`- sets the mode to back online

## Citing the paper

```text
@inproceedings{havtorn_benchmarking_2022,
	title = {Benchmarking Generative Latent Variable Models for Speech},
	journal = {Deep Generative Models for Highly Structured Data Workshop, {ICLR} 2022},
	author = {Havtorn, Jakob D. and Borgholt, Lasse and Hauberg, S{\o}ren and Frellsen, Jes and Maal{\o}e, Lars},
	year = {2022},
}
```
