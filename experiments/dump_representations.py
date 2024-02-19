"""Script to compute representations via a trained model and dump them to disk"""

import argparse
import os
import rich

from collections import defaultdict

import torch
import numpy as np

from torch.utils.data import DataLoader

from blvm.data.base_dataset import BaseDataset
from blvm.data.batchers import DynamicTensorBatcher
from blvm.data.datasets import DATASETS
from blvm.data.loaders import AudioLoader
from blvm.data.samplers.length_samplers import LengthEvalSampler
from blvm.data.transforms import Compose, MuLawDecode, MuLawEncode
from blvm.evaluation.tracker import Tracker
from blvm.settings import DATA_DIRECTORY

from blvm.utils.argparsing import float_or_str, str2bool
from blvm.utils.device import get_device, to_device_recursive
from blvm.utils.rand import get_random_seed, set_seed
from blvm.utils.wandb import get_run, restore_run
from blvm.training.restore import load_run


parser = argparse.ArgumentParser()
parser.add_argument("--entity", type=str, help="wandb entity to get the run from")
parser.add_argument("--project", type=str, default="benchmarks", help="wandb project to get the run from")
parser.add_argument("--id", type=str, help="wandb run id to get the run from")

parser.add_argument("--extra", default="s256", type=str, help="extra extension of files to save")

parser.add_argument("--dataset", type=str, default="timit", choices=DATASETS.keys(), help="dataset to use")
parser.add_argument("--sample_rate", default=16000, type=int, help="sample rate")

parser.add_argument("--num_samples", default=1, type=int, help="number of samples used to marginalize per example")
parser.add_argument("--batch_size", type=int, default=0, help="batch size in number of examples")
parser.add_argument("--batch_len", type=float_or_str, default=25, help="batch size in sequence length")

parser.add_argument("--seed", type=int, default=get_random_seed(), help="")
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="")
parser.add_argument("--use_amp", type=str2bool, default=False, help="if true, use automatic mixed precision")
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--dry", action="store_true", help="dry run without saving")

args = parser.parse_args()

if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)
rich.print(vars(args))


device = get_device() if args.device == "auto" else torch.device(args.device)
torch.cuda.set_device(device)


run = get_run(args.id, args.project, args.entity)
directory = restore_run(run, exclude="media")
model, checkpoint = load_run(directory, device=device)


dataset = DATASETS[args.dataset]
datapath = os.path.join(DATA_DIRECTORY, dataset.name)


decode_transform = []
encode_transform = []
if run.config["input_coding"] == "mu_law":
    encode_transform.append(MuLawEncode(bits=run.config["num_bits"]))
    decode_transform.append(MuLawDecode(bits=run.config["num_bits"]))

encode_transform = Compose(*encode_transform)
decode_transform = Compose(*decode_transform)

modalities = [(AudioLoader(dataset.audio_ext), encode_transform, DynamicTensorBatcher())]

dataloaders = dict()
for source_name in [*dataset.valid_sets, *dataset.test_sets, dataset.train]:
    data = BaseDataset(source=source_name, modalities=modalities)
    sampler = LengthEvalSampler(
        source=source_name,
        field=dataset.audio_length,
        shuffle=False,
        batch_size=args.batch_size * 3,
        batch_len=args.sample_rate * args.batch_len,
    )
    loader = DataLoader(
        dataset=data,
        collate_fn=data.collate,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dataloaders[source_name] = loader
    rich.print(data)
    rich.print(sampler)

print(f"Dumping representations for {dataset.name} with model {run.id}: {run.name}.")
print(f"Saving to: {datapath}.")


tracker = Tracker()

model.eval()

with torch.no_grad():
    for loader in dataloaders.values():
        for (x, x_sl), metadata in tracker.steps(loader):
            x = x.to(device, non_blocking=True)

            representations = defaultdict(lambda: [])
            lengths = defaultdict(lambda: [])
            for i in range(args.num_samples):
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    loss, metrics, output = model(x, x_sl)

                tracker.update(metrics)

                output.z = [output.z] if isinstance(output.z, torch.Tensor) else output.z
                output.z_sl = [output.z_sl] * len(output.z) if isinstance(output.z_sl, torch.Tensor) else output.z_sl
                for i, (z, z_sl) in enumerate(zip(output.z, output.z_sl)):
                    representations[i].append(z)
                    lengths[i] = z_sl

            # Compute average representation over samples (loop over hierarchy)
            for i in representations.keys():
                representations[i] = torch.mean(torch.stack(representations[i], 0), 0)

            # Extract the correct length of each example and each sample
            for i in representations.keys():  # hierarchy
                representations[i] = list(representations[i].unbind(0))
                for j in range(len(representations[i])):  # batch
                    representations[i][j] = representations[i][j][:lengths[i][j]]

            # To CPU
            for i in representations.keys():  # hierarchy
                representations[i] = to_device_recursive(representations[i], "cpu")

            # Dump to disk
            example_ids = [m.example_id for m in metadata]
            for j in range(x.size(0)):
                example_id = example_ids[j]
                for i in range(len(representations)):
                    subpath = os.path.relpath(example_ids[j], datapath)
                    path = os.path.join(datapath, subpath)
                    ext = f".{args.id}-{args.project}-{args.extra}-z{i}-n{args.num_samples}.npy"

                    if args.dry:
                        print(f"Dry run: would save to: {path}{ext}")
                    else:
                        np.save(path + ext, representations[i][j].numpy())
