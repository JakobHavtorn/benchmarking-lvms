import math

from copy import deepcopy
from typing import List, Optional, Set, Union

import editdistance
import numpy as np

import torch
import wandb

from blvm.utils.operations import detach, detach_to_device, update_running_variance


class Metric:
    base_tags = set()
    _str_value_fmt = "<.3"

    def __init__(
        self,
        name: str,
        tags: Set[str] = None,
        get_best: str = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        self.name = name
        self.tags = self.base_tags if tags is None else (tags | self.base_tags)
        self.get_best = GET_BEST[get_best] if get_best is not None else GET_BEST["none"]
        self.log_to_console = log_to_console
        self.log_to_framework = log_to_framework

    @property
    def value(self):
        """Primary value of the metric to be used for logging"""
        raise NotImplementedError()

    @property
    def str_value(self):
        return f"{self.value:{self._str_value_fmt}f}"

    def update(self, metric):
        """Update the metric (e.g. running mean)"""
        raise NotImplementedError()

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, value={self.str_value})"


def min_value(metrics: List[Metric]):
    return min(metrics, key=lambda m: m.value)


def max_value(metrics: List[Metric]):
    return max(metrics, key=lambda m: m.value)


def no_value(metrics: List[Metric]):
    return None


GET_BEST = dict(none=no_value, min=min_value, max=max_value)


class ErrorRateMetric(Metric):
    _str_value_fmt = "<.3"

    def __init__(self, refs, hyps, tokenizer, name="er", tags=None):
        super().__init__(name=name, tags=tags, get_best="min")

        rtoks = map(tokenizer, refs)
        htoks = map(tokenizer, hyps)
        self._edits, self._len = np.sum([(editdistance.eval(r, h), len(r)) for r, h in zip(rtoks, htoks)], axis=0)

    @property
    def value(self):
        return self._edits / self._len

    def update(self, metric: Metric):

        self._edits += metric._edits
        self._len += metric._len


class ConfusionMatrixMetric(Metric):
    def __init__(
        self,
        y_pred: Union[int, torch.Tensor],
        y_true: Union[int, torch.Tensor],
        class_names: List[str] = None,
        name: str = "cm",
        tags: Set[str] = None,
        log_to_framework: bool = True,
    ):
        super().__init__(name, tags, get_best=None, log_to_console=False, log_to_framework=log_to_framework)
        self.y_pred = detach_to_device(y_pred, "cpu")
        self.y_true = detach_to_device(y_true, "cpu")
        self.class_names = class_names

    @property
    def value(self):
        return wandb.plot.confusion_matrix(
            probs=None,
            y_true=self.y_true,
            preds=self.y_pred,
            class_names=self.class_names,
        )

    def update(self, metric: Metric):
        self.y_pred += metric.y_pred
        self.y_true += metric.y_true


class LatestMeanMetric(Metric):
    _str_value_fmt = "<.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: str = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Create a latest mean metric that maintains the latest mean when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to `values.numel()`.
        """
        super().__init__(
            name=name, tags=tags, get_best=get_best, log_to_console=log_to_console, log_to_framework=log_to_framework
        )

        values = detach(values)
        reduce_by = detach(reduce_by)

        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist() if isinstance(values, torch.Tensor) else values

        reduce_by = reduce_by.sum().tolist() if isinstance(reduce_by, torch.Tensor) else (reduce_by or numel)

        self.latest = value / reduce_by

    @property
    def value(self):
        return self.latest

    def update(self, metric: Metric):
        self.latest = metric.latest


class EMAMetric(Metric):
    _str_value_fmt = "<.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: str = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Create an exponential moving average of the mean metric that maintains the ema mean when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to `values.numel()`.
            weight_by (Optional[Union[torch.Tensor, float]], optional): A single or per example weights for the running mean. Defaults to `reduce_by`.
        """
        super().__init__(
            name=name, tags=tags, get_best=get_best, log_to_console=log_to_console, log_to_framework=log_to_framework
        )

        values = detach(values)
        reduce_by = detach(reduce_by)
        weight_by = detach(weight_by)

        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist() if isinstance(values, torch.Tensor) else values

        reduce_by = reduce_by.sum().tolist() if isinstance(reduce_by, torch.Tensor) else (reduce_by or numel)
        weight_by = weight_by.sum().tolist() if isinstance(weight_by, torch.Tensor) else (weight_by or reduce_by)

        self.weight_by = weight_by
        self.ema = value / reduce_by

    @property
    def value(self):
        return self.ema

    def update(self, metric: Metric):
        avg_weight = (self.weight_by + metric.weight_by) / 2  # usually these weights are equal
        self.ema = avg_weight * metric.ema + (1 - avg_weight) * self.ema


class RunningMeanMetric(Metric):
    _str_value_fmt = "<.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: str = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Create a running mean metric that maintains the running mean when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric of shape (B,)
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to `values.numel()`.
            weight_by (Optional[Union[torch.Tensor, float]], optional): A single or per example weights for the running mean. Defaults to `reduce_by`.
        """
        super().__init__(
            name=name, tags=tags, get_best=get_best, log_to_console=log_to_console, log_to_framework=log_to_framework
        )

        values = detach(values)
        reduce_by = detach(reduce_by)
        weight_by = detach(weight_by)

        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist() if isinstance(values, torch.Tensor) else values

        reduce_by = reduce_by.sum().tolist() if isinstance(reduce_by, torch.Tensor) else (reduce_by or numel)
        weight_by = weight_by.sum().tolist() if isinstance(weight_by, torch.Tensor) else (weight_by or reduce_by)

        self.weight_by = weight_by
        self.running_mean = value / reduce_by

    @property
    def value(self):
        return self.running_mean

    def update(self, metric: Metric):
        """Update the running mean statistic.

        Args:
            metric (RunningMeanMetric): The running mean metric to update with
        """
        d = self.weight_by + metric.weight_by
        w1 = self.weight_by / d
        w2 = metric.weight_by / d

        self.running_mean = self.running_mean * w1 + metric.running_mean * w2
        self.weight_by = d


class RunningVarianceMetric(Metric):
    _str_value_fmt = "<.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: str = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Create a running variance metric that maintains the running variance when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric of shape (B,)
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to `values.numel()`.
            weight_by (Optional[Union[torch.Tensor, float]], optional): A single or per example weights for the running mean. Defaults to `reduce_by`.
        """
        super().__init__(
            name=name, tags=tags, get_best=get_best, log_to_console=log_to_console, log_to_framework=log_to_framework
        )

        values = detach(values)
        reduce_by = detach(reduce_by)
        weight_by = detach(weight_by)

        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist() if isinstance(values, torch.Tensor) else values

        reduce_by = reduce_by.sum().tolist() if isinstance(reduce_by, torch.Tensor) else (reduce_by or numel)
        weight_by = weight_by.sum().tolist() if isinstance(weight_by, torch.Tensor) else (weight_by or reduce_by)

        # sum of squares of differences from the current mean
        self.weight_by = weight_by
        self.running_mean = value / reduce_by
        self.M2 = ((values - self.running_mean) ** 2).sum().item() if isinstance(values, torch.Tensor) else 0
        self.population_variance = self.M2 / (reduce_by - 1) if reduce_by > 1 else float("nan")  # unbiased variance

    @property
    def value(self):
        return self.population_variance

    def update(self, metric: Metric):
        """Update the running variance statistic.

        Args:
            metric (RunningMeanMetric): The running variance metric to update with.
        """
        var, avg, w, M2 = update_running_variance(
            mean_a=self.running_mean,
            mean_b=metric.running_mean,
            weight_a=self.weight_by,
            weight_b=metric.weight_by,
            M2_a=self.M2,
            M2_b=metric.M2,
        )
        self.running_mean = avg
        self.population_variance = var
        self.weight_by = w
        self.M2 = M2


class RunnnigAccuracyMetric(Metric):
    _str_value_fmt = "6.4"

    def __init__(
        self,
        predictions: Union[torch.Tensor, float],
        labels: Union[torch.Tensor, float],
        name: str = "accuracy",
        tags: Set[str] = None,
        get_best: float = "max",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Standard classification accuracy"""
        super().__init__(
            name=name, tags=tags, get_best=get_best, log_to_console=log_to_console, log_to_framework=log_to_framework
        )
        predictions = detach(predictions)
        labels = detach(labels)
        self.correct = (predictions == labels).sum().item()
        self.total = labels.size(0)

    @property
    def value(self):
        return self.correct / self.total

    def update(self, metric: Metric):
        self.correct += metric.correct
        self.total += metric.total


class LossMetric(RunningMeanMetric):
    base_tags = {"losses"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "loss",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: float = "min",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class LLMetric(RunningMeanMetric):
    base_tags = {"log_likelihoods"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "ll",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: float = "max",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class KLMetric(RunningMeanMetric):
    base_tags = {"kl_divergences"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "kl",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: float = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class BitsPerDimMetric(RunningMeanMetric):
    base_tags = set()
    _str_value_fmt = "<5.3"  # 5.321

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "bpd",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: float = "min",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        values = -detach(values) / math.log(2)
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class PerplexityMetric(BitsPerDimMetric):
    """Perplexity computed as $2^{-\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)}$"""

    base_tags = set()
    _str_value_fmt = "<8.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "pp",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: float = "min",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )

    @property
    def value(self):
        return 2 ** self.running_mean
