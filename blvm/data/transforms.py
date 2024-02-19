import math

from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from torchaudio.transforms import MelSpectrogram

from blvm.utils.operations import stack_tensor


class Transform(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError()

    def __repr__(self):
        name = self.__class__.__name__
        attrs = vars(self)
        var_str = ", ".join([f"{k}={v}" for k, v in attrs.items() if k[0] != "_" and k != "training"])
        return f"{name}({var_str})"


class Compose(Transform):
    def __init__(self, *transforms: List[Transform]):
        self.transforms = [transform for transform in transforms if transform is not None]

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self):
        format_strings = []
        for t in self.transforms:
            format_strings.append(str(t))

        if len(", ".join(format_strings)) < 110:
            s = ", ".join(format_strings)
            end = ")"
        else:
            s = "\n    " + ",\n    ".join(format_strings)
            end = "\n)"

        s = self.__class__.__name__ + "(" + s + end
        return s


class TextCleaner(Transform):
    def __init__(self, *cleaner_fcns: List[Callable]):
        super().__init__()
        self.cleaner_fcns = cleaner_fcns

    def forward(self, x: str):
        for fcn in self.cleaner_fcns:
            x = fcn(x)
        return x


class EncodeInteger(Transform):
    def __init__(self, tokenizer, token_map):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_map = token_map

    def forward(self, x: str):
        x = self.tokenizer(x)
        x = self.token_map.encode(x)
        return x


class DecodeInteger(Transform):
    def __init__(self, join_token, token_map):
        super().__init__()
        self.join_token = join_token
        self.token_map = token_map

    def forward(self, x: str):
        x = self.token_map.decode(x)
        x = self.join_token.join(x)
        return x


class StackTensor(Transform):
    def __init__(self, n_frames: int, dim=-1):
        super().__init__()
        self.n_frames = n_frames
        self.dim = dim

    def forward(self, x):
        x, _ = stack_tensor(x, self.n_frames, dim=self.dim)
        return x


class RandomSegment(Transform):
    def __init__(self, length: int):
        """Randomly sample a segment of a certain length from an example of dimensions (T, *)"""
        super().__init__()
        self.length = length

    def forward(self, x):
        high = max(x.size(0) - self.length, 1)
        start_idx = torch.randint(low=0, high=high, size=(1,))
        return x[start_idx : start_idx + self.length]


class LogMelSpectrogram(Transform):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: int = 80,
        normalize_frq_bins: bool = True
    ) -> None:
        """Mel spectrogram in log space from audio waveform.

        Compared to PyTorch MelSpectrogram transform, normalization is applied per frequency bin.

        Args:
            sample_rate (int, optional): [description]. Defaults to 16000.
            n_fft (int, optional): [description]. Defaults to 400.
            win_length (Optional[int], optional): [description]. Defaults to None.
            hop_length (Optional[int], optional): [description]. Defaults to None.
            n_mels (int, optional): [description]. Defaults to 80.
            normalize_frq_bins (bool, optional): [description]. Defaults to True.
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize_frq_bins = normalize_frq_bins

        self.MelSpectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).
        Returns:
            Tensor: specgram_mel_db of size (..., ``n_mels``, time).
        """
        mel_specgram = self.MelSpectrogram(waveform)
        logmel_specgram = 10 * torch.log10(torch.clamp_min(mel_specgram, 1e-10))
        
        if self.normalize_frq_bins:
            logmel_specgram -= torch.mean(logmel_specgram, -1, keepdim=True)
            logmel_specgram /= torch.std(logmel_specgram, -1, keepdim=True) + 1e-10

        return logmel_specgram


class Normalize(Transform):
    def __init__(self, mean: Union[float, torch.Tensor] = None, std: Union[float, torch.Tensor] = None, dim: int = -1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.dim = dim

    def forward(self, x):
        mean = x.mean(self.dim) if self.mean is None else self.mean
        std = x.std(self.dim) if self.std is None else self.std
        return (x - mean) / std


class Denormalize(Transform):
    def __init__(self, mean: Union[float, torch.Tensor] = None, std: Union[float, torch.Tensor] = None):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return x * self.std + self.mean


class MuLawEncode(Transform):
    def __init__(self, bits: int = 8):
        """Encode PCM audio in [-1, 1] via µ-law companding to some number of bits (8 by default)"""
        super().__init__()
        self.bits = bits
        self.mu = 2 ** bits - 1
        self._divisor = math.log(self.mu + 1)

    def forward(self, x: torch.Tensor):
        return x.sign() * torch.log(1 + self.mu * x.abs()) / self._divisor


class MuLawDecode(Transform):
    def __init__(self, bits: int = 8):
        """Decode PCM (µ-law encoded) audio in [-1, 1] via µ-law companding from some number of bits (8 by default)"""
        super().__init__()
        self.bits = bits
        self.mu = 2 ** bits - 1
        self._divisor = math.log(self.mu + 1)

    def forward(self, x: torch.Tensor):
        return x.sign() * (torch.exp(x.abs() * self._divisor) - 1) / self.mu


class Quantize(Transform):
    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        bits: int = 8,
        bins: Optional[int] = None,
        force_out_int64: bool = True,
        rescale: bool = False,
    ):
        """Quantize a tensor of values between `low` and `high` using a number of `bits`.

        The return value is an integer tensor with integer values in [0, 2**bits - 1], if rescale == False.
        The return value is rescaled to floats in [low, high] if rescale == True.

        If `bits` is 32 or smaller, the integer tensor is of type `IntTensor` (32 bits).
        If `bits` is 33 or larger, the integer tensor is of type `LongTensor` (64 bits).

        We can force `LongTensor` (64 bit) output if `force_out_int64` is `True`.

        Args:
            low (float, optional): [description]. Defaults to -1.0.
            high (float, optional): [description]. Defaults to 1.0.
            bits (int, optional): [description]. Defaults to 8.
            bins (Optional[int], optional): [description]. Defaults to None.
            force_out_int64 (bool): If False and bits <= 32, will output int32. Otherwise output is int64.
            rescale (bool): If True, rescale quantized integer values back to floats in [low, high].
        """
        super().__init__()
        assert (bits is None) != (bins is None), "Must set one and only one of `bits` and `bins`"
        self.low = low
        self.high = high
        self.bits = bins // 8 if bits is None else bits
        self.bins = 2 ** bits if bins is None else bins
        self.boundaries = torch.linspace(start=-1, end=1, steps=self.bins)
        self.out_int32 = (self.bits <= 32) and (not force_out_int64)
        if rescale:
            self.rescale = Scale(low=low, high=high, min_val=0, max_val=self.bins -1)
        else:
            self.rescale = None

    def forward(self, x: torch.Tensor):
        x_quantized = torch.bucketize(x, self.boundaries, out_int32=self.out_int32, right=False)
        x_quantized = self.rescale(x_quantized) if self.rescale is not None else x_quantized
        return x_quantized
