from typing import List, Tuple, Any, Optional

import torch


class Batcher:
    """Base class for Batchers. These must define `collate` and optionally `sort` methods."""

    def __init__(self) -> None:
        pass

    def __call__(self, batch: List[torch.Tensor]):
        return self.collate(batch)

    def collate(self, batch: List[torch.Tensor]):
        """Convert a list of Tensors into a single Tensor.

        Args:
            batch (List[torch.Tensor]): Batch of Tensors to convert.
        """
        raise NotImplementedError()

    def sort(self, batch: List[Tuple[Any, Any]], sort_modality_idx: Optional[int] = None):
        """Sort the order of examples within the batch optionally specifying which modality to sort if more than one.

        Args:
            batch (List[Tuple[Any, Any]]): The batch to sort, generally as a list of tuples of data and metadata.
            sort_modality_idx (bool, optional): Index of the modality to sort. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ListBatcher(Batcher):
    def __init__(self) -> None:
        """Generic batcher that simply returns a list of any object and tries to infer their lengths.
        
        Infers length with torch.numel() for torch.Tensor.
        Infers length with len() method for objects with __len__. 
        Returns 0 as length if neither is applicable.
        """
        super().__init__()

    def collate(self, batch: List[Any]):
        if isinstance(batch[0], torch.Tensor):
            sequence_lengths = [tensor.numel() for tensor in batch]
        elif hasattr(batch[0], "__len__"):
            sequence_lengths = [len(element) for element in batch]
        else:
            sequence_lengths = [0 for element in batch]
        return batch, torch.LongTensor(sequence_lengths)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: len(x[0][sort_modality_idx])
        else:
            sort_key = lambda x: len(x[0])

        return sorted(batch, key=sort_key, reverse=True)


class TextBatcher(Batcher):
    def __init__(self, pad_value: int = 0) -> None:
        self.pad_value = pad_value

    def collate(self, batch: List[List[int]]):
        """Pad batch of int (encoded and tokenized text) to maximum temporal length and return LongTensors"""
        sequence_lengths = [len(text) for text in batch]

        T = max(sequence_lengths)

        collated_batch = []
        for t, text in zip(sequence_lengths, batch):
            collated_batch.append(text + [self.pad_value] * (T - t))

        return torch.LongTensor(collated_batch), torch.LongTensor(sequence_lengths)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: len(x[0][sort_modality_idx])
        else:
            sort_key = lambda x: len(x[0])

        return sorted(batch, key=sort_key, reverse=True)


class TensorBatcher(Batcher):
    def __init__(self):
        """Generic concatenating batcher for equally sized tensors of arbitrary dimensions"""
        super().__init__()

    def collate(self, batch: List[torch.Tensor]):
        """Concatenate a number of equally sized tensors (B, D1, D2, D3, ...). Sequence length is `tensor.numel()`"""
        sequence_lengths = [tensor.numel() for tensor in batch]
        shapes = [tensor.shape for tensor in batch]

        assert all(sequence_lengths[0] == seq_len for seq_len in sequence_lengths)
        assert all(shapes[0] == shape for shape in shapes)

        collated_batch = torch.cat(batch, dim=0)

        return collated_batch, torch.LongTensor(sequence_lengths)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        return batch


class DynamicTensorBatcher(Batcher):
    def __init__(self, dim: int = -1, pad_value: float = 0) -> None:
        """Batcher that pads tensors to maximum length along a single dynamic dimemsion `dim` and concatenates them."""
        super().__init__()
        self.dim = dim
        self.pad_value = pad_value

    def collate(self, batch: List[torch.Tensor]):
        """Zero pad batch of tensors of some shape (B, *, D, *) to maximum temporal length and concatenate"""
        sequence_lengths = [tensor.shape[self.dim] for tensor in batch]

        N = len(batch)
        T = max(sequence_lengths)

        # get shape of batch with full temporal dimension
        collated_shape = list(batch[0].shape)
        collated_shape[self.dim] = T
        collated_shape = [N] + collated_shape  # (B, *, D, *)

        # move padding dimension to end to allow easy indexing into collated_batch below
        collated_shape[self.dim], collated_shape[-1] = collated_shape[-1], collated_shape[self.dim]  # (B, *, D)

        dtype = batch[0].dtype
        collated_batch = torch.full(collated_shape, dtype=dtype, fill_value=self.pad_value)  # (B, *, D)
        for i, seq_len in enumerate(sequence_lengths):
            collated_batch[i, ..., :seq_len] = batch[i].transpose(self.dim, -1)

        # revert transpose
        collated_batch = collated_batch.transpose(self.dim, -1)  # (B, *, D, *)

        return collated_batch, torch.LongTensor(sequence_lengths)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: x[0][sort_modality_idx].shape[self.dim]
        else:
            sort_key = lambda x: len(x[0])

        return sorted(batch, key=sort_key, reverse=True)
