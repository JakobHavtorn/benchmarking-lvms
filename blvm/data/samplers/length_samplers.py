import random

from typing import Iterator, Optional, Union, List

import more_itertools as mit
import numpy as np

from torch.utils.data.sampler import Sampler

from .utils import load_field
from ..base_dataset import BaseDataset


def parse_max_len(batch_len: Union[int, float, str], lengths: List[int]):
    """Parse the `batch_len` argument with error checking

    Args:
        batch_len (Union[int, float, str]): The length of the batch to return. If `int` or `float`, the batch will be of
                                            length `batch_len` time steps. If `str`, must contain the substring `'max'`.
                                            The batch will then be of length `d * max(lengths)` time steps where `d` is
                                            any digit prepended to `'max'`.
        lengths: The lengths of the examples in the dataset.

    Returns:
        float: Batch length in time steps (e.g. seconds).
    """
    if batch_len is None:
        raise ValueError(f"`batch_len` cannot be `None`")

    max_len = max(lengths)
    if isinstance(batch_len, int) or isinstance(batch_len, float):
        if batch_len < max_len:
            raise ValueError(f"Given `batch_len` shorter than longest example {max_len}, would create empty batches.")
        return batch_len

    if isinstance(batch_len, str):
        if not "max" in batch_len:
            raise ValueError(f"`batch_len` must be `int`, `float`, or contain the substring `'max'`")

        digits = [c for c in batch_len if c.isdigit()]
        if not digits:
            return max_len
        return int("".join(digits)) * max_len  # e.g. "4max" or "4 * max" gives "bathc_len = 4 * max(lengths)"

    raise ValueError(f"`batch_len` must be an integer, float, or 'max'")


class LengthTrainSampler(Sampler):
    def __init__(
        self,
        source: Union[str, BaseDataset, List[int]],
        field: Optional[str] = "length",
        max_pool_difference: Optional[float] = None,  # 16K * 0.3
        min_pool_size: int = 512,
        batch_len: Optional[Union[float, str]] = None,
        batch_size: Optional[Union[float, str]] = None,
        num_batches: Optional[int] = None,
        shuffle: bool = True,
        longest_first: bool = True,
        drop_last: bool = True,
    ):
        """
        This batch_sampler groups the source into sample pools of examples with similar length meeting criterias defined
        by 'max_pool_difference' and 'min_pool_size'. Batches of close to, but never more than, 'batch_len', are
        constructed by first sampling a pool and then sampling each batch from from within that pool.

        Args:
            source (object): Source for sampling. Either:
                             1) a source file name (`str`) of a dataset with a column called `field` of example lengths,
                             2) a BaseDataset used to obtain the example lengths via looping over it,
                             3) a list of integer lengths used directly.
            field (str): The field containing the relevant length information in the souce file.
            batch_len (float): Maximum length of the batch in the unit of lengths. If "max", set to length of longest.
            batch_size (int): Maximum size of the batch in number of examples. Useful if e.g. accumulating gradient.
            max_pool_difference (float): The maximum length difference between shortest and longest sample a pool.
            min_pool_size (float): The minimum number of examples in a pool. Overwrites max_pool_difference.
            num_batches (int or None): Samples num_batches (with replacement if necessary) instead of running an epoch.
            shuffle (bool): If True, sample new batches on every epoch. If False, sample batches once. Defaults to True.
            longest_first (bool): If True, returns the longest batch first and then the rest in (random) order.
                                  This happens only on the first epoch and helps memory allocation. Defaults to True.
            drop_last (bool): If True, drops the last batch if it is smaller than batch_size. Defaults to True.
        """
        assert sum([bool(batch_len), bool(batch_size)]) == 1, "batch_len and batch_size are mutually exclusive."

        self.source = source
        self.field = field
        self.max_pool_difference = max_pool_difference
        self.min_pool_size = min_pool_size
        self.batch_size = batch_size
        self.batch_len = batch_len
        self.num_batches = num_batches
        self.shuffle = shuffle
        self.longest_first = longest_first
        self.buffer = []  # only used when num_batches is not None
        self.drop_last = drop_last

        lengths = source if isinstance(source, list) else load_field(source, field)
        self.lengths = np.asarray(lengths, dtype=int)

        if max_pool_difference is None:
            max_pool_difference = (max(self.lengths) - min(self.lengths)) * 0.05

        self.sorted_indices = np.argsort(self.lengths)

        if batch_len:
            self.batch_len = parse_max_len(batch_len, self.lengths)
        else:
            raise NotImplementedError(f"`batch_size` is not yet implemented.")

        self.pools = self.create_sample_pools(max_pool_difference, min_pool_size)

        self.sample_batches()

        if self.longest_first:
            self.move_longest_to_front()

    def move_longest_to_front(self):
        batch_lengths = [max([self.lengths[b] for b in batch]) for batch in self.batches]
        max_idx = batch_lengths.index(max(batch_lengths))
        self.batches[0], self.batches[max_idx] = self.batches[max_idx], self.batches[0]

    def create_sample_pools(self, max_diff, min_size):
        """Creates the sample pools. Can be used to change to the sampling criteria without creating a new sampler."""
        start, end = 0, 0
        sorted_lens = self.lengths[self.sorted_indices]

        pools = []
        while end != len(self.lengths):
            base_len = sorted_lens[start]
            deltas = sorted_lens - base_len
            pool_size = np.logical_and(0 <= deltas, deltas < max_diff).sum()
            end = min(max(start + min_size, start + pool_size), len(self.lengths))
            if (len(self.lengths) - end) < min_size:
                end = len(self.lengths)

            pools.append(self.sorted_indices[start:end].tolist())
            start = end

        return pools

    def sample_batches(self):
        """Sample batches from the pools."""
        if self.num_batches is not None:
            if len(self.buffer) >= self.num_batches:
                self.batches = self.buffer[: self.num_batches]
                self.buffer = self.buffer[self.num_batches :]
                return None

        ordered_idxs = np.concatenate([random.sample(p, k=len(p)) for p in self.pools])  # shuffle each pool internally

        batch, batches, batch_len = [], [], 0
        for idx in ordered_idxs:
            l = self.lengths[idx]
            if batch_len + l <= self.batch_len:
                batch_len += l
                batch.append(idx)
            else:
                batches.append(batch)
                batch = [idx]
                batch_len = l
        if batch and not (self.drop_last and batch_len < self.batch_len):
            # add last batch if any and not dropping last (we also only drop last if it is shorter than batch_len)
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)  # shuffle the order of batches

        if self.num_batches is not None:
            self.buffer += batches
            self.sample_batches()

        self.batches = batches

    def __iter__(self) -> Iterator[List[int]]:
        try:
            for batch in self.batches:
                yield batch
        finally:
            if self.shuffle:
                self.sample_batches()  # to ensure batches are resampled if interrupted

    def __len__(self):
        return len(self.batches)

    def __repr__(self):
        source = self.source
        field = self.field
        batch_len = self.batch_len
        batch_size = self.batch_size
        max_pool_difference = self.max_pool_difference
        min_pool_size = self.min_pool_size
        num_batches = self.num_batches
        s = f"LengthTrainSampler({source=}, {field=}, {batch_size=}, {batch_len=}, {max_pool_difference=}, {min_pool_size=}, {num_batches=})"
        return s


class LengthEvalSampler(Sampler):
    def __init__(
        self,
        source: Union[str, BaseDataset, List[int]],
        field: Optional[str] = "length",
        batch_len: Optional[float] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        longest_first: bool = True,
    ):
        """
        This batch_sampler groups the source into sample pools of examples with similar length meeting criterias defined
        by 'max_pool_difference' and 'min_pool_size'. Batches of up to 'seconds' are constructed by sampling from
        one pool at the time.

        Args:
            source (object): Dataset for which the sampler will be used.
            field (str): name of column with file lengths in source file.
            batch_len (float): Max size of a batch in seconds.
            batch_size (int): Max size of a batch in examples. Useful if evaluating on split examples of equal length.
            shuffle (bool): If True, shuffle the order of the batches between epochs (the batches are always the same).
                            Otherwise, batches are presented in order from those containing the shortest examples to
                            those containing the longest if `longest_first` is False, opposite otherwise.
                            Defaults to False.
            longest_first (bool): If True, batches are returned from longest to shortest. If `shuffle` is True, this
                                  happens only on the first epoch. Defaults to True.
        """
        assert sum([bool(batch_len), bool(batch_size)]) == 1, "batch_len and batch_size are mutually exclusive."

        self.source = source
        self.field = field
        self.batch_size = batch_size
        self.batch_len = batch_len
        self.shuffle = shuffle
        self.longest_first = longest_first

        lengths = source if isinstance(source, list) else load_field(source, field)
        self.lengths = np.asarray(lengths, dtype=int)

        self.sorted_indices = np.argsort(self.lengths)

        if batch_len:
            self.batch_len = parse_max_len(batch_len, self.lengths)
            self.sample_batches = self.sample_batches_len
        else:
            self.sample_batches = self.sample_batches_size

        self.sample_batches()

    def sample_batches_size(self):
        """Create batches according to size with a fixed number of examples per batch."""
        batches = list(mit.chunked(self.sorted_indices, self.batch_size))

        if self.longest_first:
            # disable after first, if shuffling. If not shuffing, the longest is first for all epochs.
            self.longest_first = not self.shuffle
            batches.reverse()
        elif self.shuffle:
            random.shuffle(batches)

        self.batches = batches

    def sample_batches_len(self):
        """Create batches according to length with a approximate minimum total length of examples per batch."""
        batch, batches, batch_len = [], [], 0
        for idx in self.sorted_indices:
            l = self.lengths[idx]
            if batch_len + l <= self.batch_len:
                batch_len += l
                batch.append(idx)
            else:
                batches.append(batch)
                batch = [idx]
                batch_len = l
        if batch:
            # add last batch if any
            batches.append(batch)

        if self.longest_first:
            self.longest_first = not self.shuffle  # disable after first if shuffling
            batches.reverse()
        elif self.shuffle:
            random.shuffle(batches)

        self.batches = batches

    def __iter__(self) -> Iterator[List[int]]:
        try:
            for batch in self.batches:
                yield batch
        finally:
            if self.shuffle:
                self.sample_batches()  # to ensure batches are resampled if interrupted

    def __len__(self):
        return len(self.batches)

    def __repr__(self):
        source = self.source
        field = self.field
        batch_len = self.batch_len
        batch_size = self.batch_size
        s = f"LengthEvalSampler({source=}, {field=}, {batch_size=}, {batch_len=})"
        return s
