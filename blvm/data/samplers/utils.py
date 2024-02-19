import csv
import logging
import os

from typing import Union, List

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_dataset import BaseDataset
from ..datapaths import DATAPATHS_MAPPING


LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)


def load_field(source: Union[str, BaseDataset], field: Union[str, List[str]], chunk_size: int = 1000, num_workers: int = None):
    """Get values by trying to read source file. Falls back to iterating over dataset if source is a dataset."""
    source_string = source if isinstance(source, str) else source.source
    source_filepath = DATAPATHS_MAPPING[source_string] if source_string in DATAPATHS_MAPPING else source_string
    try:
        values = load_field_from_sourcefile(source_filepath, field)
    except KeyError as exc:
        if isinstance(source, str):
            raise exc
        LOGGER.info("Failed to load values from source file with error: " + str(exc))
        values = load_field_from_dataset(source, field, chunk_size=chunk_size, num_workers=num_workers)
    return values


def load_field_from_sourcefile(source_filepath: str, field: Union[str, List[str]]):
    """Loads the example values from source file into an array with same order as the examples of the source dataset."""
    if isinstance(field, str):
        get_value = lambda row: row[field]
    elif isinstance(field, list):
        get_value = lambda row: [row[f] for f in field]
    else:
        raise ValueError(f"`field` must be a string or list of strings, not {type(field)}")

    with open(source_filepath, newline="") as source_file_buffer:
        reader = csv.DictReader(source_file_buffer)
        try:
            values = []
            for row in reader:
                values.append(get_value(row))
        except KeyError:
            raise KeyError(f"`{field}` not in columns {list(row.keys())} of file {source_filepath}")

    return values


def load_field_from_dataset(source: BaseDataset, field: Union[str, List[str]], chunk_size: int = 1000, num_workers: int = None):
    """Loads the field from the source dataset into a list with same order as the examples of the source dataset.

    Args:
        source (BaseDataset): The source dataset.
        field (str): The field to load.
        chunk_size (int, optional): Chunks loaded by each process per iteration. Defaults to 1000.
        num_workers (int, optional): Number of workers to use for loading. Defaults to `os.cpu_count() // 2`.
    """
    if isinstance(field, str):
        get_value = lambda metadata: getattr(metadata, field)
    elif isinstance(field, list):
        get_value = lambda metadata: [getattr(metadata, f) for f in field]
    else:
        raise ValueError(f"`field` must be a string or list of strings, not {type(field)}")

    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    sampler = torch.utils.data.SequentialSampler(source)
    dataloader = DataLoader(
        source,
        batch_size=chunk_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=source.collate,
    )

    values = []
    for _, metadatas in tqdm(dataloader, desc=f"Gathering {field} from {source.source}"):
        values.extend([get_value(metadata) for metadata in metadatas])

    return values
