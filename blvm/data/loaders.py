import csv
import datetime
import os
import re
import uuid

from dataclasses import dataclass
from typing import Callable, Optional, Union, List

import numpy as np
import torch
import torchaudio

from blvm.data.datapaths import TIMIT
from blvm.data.datasets import DATASETS


@dataclass
class MetaData:
    length: int = None
    file_path: str = None
    example_id: str = None
    sub_id: int = None  # sub id for batched datasets


@dataclass
class AudioMetaData(MetaData):
    sample_rate: int = None
    channels: int = None
    bits_per_sample: int = None
    encoding: str = None


@dataclass
class TextMetaData(MetaData):
    word_length: int = None
    char_length: int = None


def load_text(file_path: str):
    with open(file_path, "r") as text_file:
        text = text_file.read()

    metadata = TextMetaData(length=len(text), char_length=len(text), word_length=len(text.split()), file_path=file_path)
    return text, metadata


def load_audio(file_path: str, sum_channels: bool = False):
    metadata = torchaudio.info(file_path)
    audio, _ = torchaudio.load(file_path)

    if sum_channels:
        audio = audio.sum(axis=0)

    metadata = AudioMetaData(
        sample_rate=metadata.sample_rate,
        channels=metadata.num_channels,
        bits_per_sample=metadata.bits_per_sample,
        encoding=metadata.encoding,
        length=metadata.num_frames,
        file_path=file_path,
    )
    return audio, metadata


def load_numpy(file_path: str, length_dim: int = 0, key: Optional[str] = None, dtype: torch.dtype = None, **kwargs):
    """Load a npy or npz file (npz requires `key` to return torch.Tensor, otherwise returns a `NpzFile`)"""
    array = np.load(file_path, **kwargs)

    if isinstance(array, np.ndarray):
        output = torch.from_numpy(array)
        length = output.size(length_dim)
    elif key is not None:
        if array[key].dtype != np.dtype("O"):
            output = torch.from_numpy(array[key])
            length = output.size(length_dim)
        else:
            # objects in key
            output = [torch.from_numpy(arr) for arr in array[key]]
            length = [o.size(length_dim) for o in output]
    else:
        output, length = array, None

    if dtype:
        output = output.to(dtype)

    metadata = MetaData(length=length, file_path=file_path)
    return output, metadata


def memoize(func: Callable):
    """Decoraator that augments a function with a dynamic memory cache that returns quickly for seen inputs"""
    cache = dict()

    def memoized_func(example_id):
        if example_id in cache:
            return cache[example_id]
        result = func(example_id)
        cache[example_id] = result
        return result

    memoized_func.memory = cache
    return memoized_func


class Loader:
    def __init__(self, extension: Union[None, str], cache: bool = False):
        """
        Base Loader for any data type.

        Args:
            extension (str): Extension of data files without delimiter.
            cache (bool): Whether to enable caching.
        """
        self.extension = extension
        self.cache = False

        self.suffix = f"{os.extsep}{extension}" if extension is not None else ""
        self.id = str(uuid.uuid4())
        self.cached_files = set()

        if cache:
            self.enable_cache()

    def enable_cache(self):
        """Enables caching for the loader."""
        if not self.cache:
            self.cache = True
            self.load = memoize(self.load)

    def __call__(self, example_id):
        """Calls the potentially memoized load method."""
        return self.load(example_id)

    def load(self, example_id):
        raise NotImplementedError

    def __repr__(self):
        name = self.__class__.__name__
        extension = self.extension
        cache = self.cache
        return f"{name}({extension=}, {cache=}, id={self.id})"


class AudioLoader(Loader):
    def __init__(self, extension: Union[None, str], cache: bool = False, sum_channels: bool = True):
        """
        Loader for audio data.

        Args:
            extension (str): Extension of data files (e.g., "wav" or "flac").
            cache (bool): Whether to enable caching.
        """
        super().__init__(extension=extension, cache=cache)
        self.sum_channels = sum_channels

    def load(self, example_id):
        """Load a single audio file."""
        file_path = example_id + self.suffix
        audio, metadata = load_audio(file_path, self.sum_channels)
        metadata.example_id = example_id
        return audio, metadata


class TextLoader(Loader):
    def __init__(self, extension: Union[None, str], cache: bool = False):
        """
        Loader for text data.

        Args:
            extension (str): Extension of data files (e.g., "txt").
            cache (bool): Whether to enable caching.
        """
        super().__init__(extension=extension, cache=cache)

    def load(self, example_id):
        """Load a single text file"""
        file_path = example_id + self.suffix
        text, metadata = load_text(file_path)
        metadata.example_id = example_id
        return text, metadata

    def load_and_cache_batch(self, batch_id):
        """Load a text file with multiple examples and cache them."""
        if not self.cache:
            raise ValueError("Caching not enabled for loader.")

        file_path = batch_id + self.suffix
        if file_path in self.cached_files:
            return

        with open(file_path, "r") as text_file:
            strings = text_file.read().splitlines()

        batch_data = {}
        for idx, string in enumerate(strings):
            example_id = f"{batch_id}-{idx}"
            metadata = TextMetaData(
                length=len(string),
                char_length=len(string),
                word_length=len(string.split()),
                example_id=example_id,
                file_path=file_path,
                sub_id=idx,
            )
            batch_data[example_id] = (string, metadata)

        self.load.memory.update(batch_data)
        self.cached_files.add(file_path)


class NumpyLoader(Loader):
    def __init__(
        self, extension: Union[None, str], cache: bool = False, length_dim: int = 0, key: Optional[str] = None, dtype: torch.dtype = None, **kwargs
    ):
        """
        Loader for numpy data.

        Args:
            extension (str): Extension of data files, npy or npz. If npz, also specify the `key` within the file to use.
            cache (bool): Whether to enable caching.
            length_dim (int): Dimension in loaded array to regard as sample length.
            kwargs (dict): key-word arguments for `numpy.load`.
        """
        super().__init__(extension=extension, cache=cache)
        self.length_dim = length_dim
        self.key = key
        self.dtype = dtype
        self.kwargs = kwargs

    def load(self, example_id):
        """Load a single audio file."""
        file_path = example_id + self.suffix
        tensor, metadata = load_numpy(file_path, self.length_dim, self.key, self.dtype, **self.kwargs)
        metadata.example_id = example_id
        return tensor, metadata

    def load_and_cache_batch(self, batch_id):
        """Load a numpy file with multiple examples and cache them."""
        if not self.cache:
            raise ValueError("Caching not enabled for loader.")

        file_path = batch_id + self.suffix
        if file_path in self.cached_files:
            return None

        tensors, metadata = load_numpy(file_path, self.length_dim, self.key, self.dtype, **self.kwargs)

        batch_data = {}
        for idx in range(len(tensors)):
            example_id = f"{batch_id}-{idx}"

            tensor = tensors[idx]
            metadata = MetaData(
                length=tensor.shape[self.length_dim], example_id=example_id, file_path=file_path, sub_id=idx
            )
            batch_data[example_id] = (tensor, metadata)

        self.load.memory.update(batch_data)
        self.cached_files.add(file_path)
