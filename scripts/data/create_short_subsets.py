"""
A script to create random subsets of dataset source files according to some specified length.

Example: Create 1h and 10m subsets of TIMIT training data.
> python scripts/data/create_short_subsets.py --dataset "timit" --source "train.txt" --lengths 57600000 9600000 --names "train_1h" "train_10m"
"""

import argparse
import csv
import os

from blvm.settings import SOURCE_DIRECTORY
from blvm.utils.rand import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", type=str, required=True, help="name of dataset to subsample e.g. `timit`")
parser.add_argument("--source", "-s", type=str, required=True, help="source file to subsample e.g. `train.txt`")
parser.add_argument("--lengths", "-l", type=int, nargs="+", required=True, help="lengths of the dataset subsets")
parser.add_argument("--length_column", type=str, default="length.flac.samples", help="name of length column in source")
parser.add_argument("--names", "-n", type=str, nargs="+", required=True, help="names of the dataset subsets")
parser.add_argument("--seed", type=int, default=0, help="seed for random sampling")
parser.add_argument("--dry", action="store_true", help="make a dry run")

args = parser.parse_args()

source_dir = os.path.join(SOURCE_DIRECTORY, args.dataset)
source_file_path = os.path.join(source_dir, args.source)

set_seed(args.seed)

# check args
if not os.path.exists(source_file_path):
    raise FileExistsError(f"Path does not exist {source_file_path}")

if not len(args.lengths) == len(args.names):
    raise ValueError(f"Must give as many wanted lengths as names but got {len(args.lengths)} and {len(args.names)}")

print(f"Creating training subsets {args.names} for {args.dataset} with lengths {args.lengths}.")

# add source file extensions if not already there
source_ext = os.extsep.join(source_file_path.split(os.extsep)[1:])
args.names = [name + os.extsep +  source_ext for name in args.names if not os.extsep in name]

# read source file as csv rows and raw file lines
with open(source_file_path, newline="") as buffer:
    reader = csv.DictReader(buffer)
    rows = list(reader)
with open(source_file_path, newline="") as buffer:
    lines = buffer.readlines()

# shuffle rows and lines randomly (but seeded)
indices = list(range(len(rows)))
rows = [rows[i] for i in indices]
lines = [lines[i] for i in indices]

# create subsets of training set by counting lengths
train_subset_lines = dict()
for name, length in zip(args.names, args.lengths):
    i, cum_length = 0, 0

    while cum_length < length:
        cum_length += int(rows[i][args.length_column])
        i += 1

    train_subset_lines[name] = lines[:i]

# save new source files
for name, subset_lines in train_subset_lines.items():
    subset_source_file_path = os.path.join(source_dir, name)

    if os.path.exists(subset_source_file_path):
        raise FileExistsError(f"Subset source file already exists: {subset_source_file_path}")

    print(f"{name:20s}: {len(subset_lines):8d} files at {subset_source_file_path}")
    if not args.dry:
        with open(subset_source_file_path, "w") as buffer:
            buffer.write( "\n".join(subset_lines))
