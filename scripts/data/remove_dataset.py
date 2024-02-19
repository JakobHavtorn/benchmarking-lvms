"""
Remove a dataset (source and data) by name.

Example: Remove TIMIT
> python remove_dataset timit
"""

import os
import sys
import shutil

from blvm.settings import DATA_DIRECTORY, SOURCE_DIRECTORY

dataset = sys.argv[1]

assert isinstance(dataset, str) and len(dataset) > 0

data_dir = os.path.join(DATA_DIRECTORY, dataset)
source_dir = os.path.join(SOURCE_DIRECTORY, dataset)

print(f"Removing dataset {dataset} with source and data directories:")
print(f"Source: {source_dir}")
print(f"Data:   {data_dir}")

assert os.path.exists(data_dir), f"Dataset {dataset} does not exist at data directory {data_dir}."
shutil.rmtree(data_dir)

assert os.path.exists(source_dir), f"Dataset {dataset} does not exist at sourcefile directory {source_dir}."
shutil.rmtree(source_dir)
