import collections
import itertools
from blessed import Terminal
import psutil
import re

from collections import defaultdict
from datetime import datetime
from time import time
from typing import Dict, Iterable, Union, Any, List, Optional, overload

import rich
import wandb

from torch.utils.data import DataLoader

from .metrics import Metric


TRACKER_STR = "tracker.pt"

FORMATTING_PATTERN = r"\[([^\]]+)\]"
FORMATTING_REGEX = re.compile(FORMATTING_PATTERN)


def length_of_formatting(string: str):
    return sum(len(s) + 2 for s in FORMATTING_REGEX.findall(string))  # plus 2 for parenthesis


def length_without_formatting(string: str):
    return len(string) - length_of_formatting(string)


def source_string(source):
    return f"{source[:18]}.." if len(source) > 20 else f"{source}"


def rank_string(rank):
    return f"[grey30]rank {rank:2d}[/grey30]"


def timestamp_string(timestamp: Optional[datetime] = None):
    if timestamp is None:
        timestamp = datetime.now()
    return "[grey30]" + timestamp.strftime("%d/%m/%Y %H:%M:%S") + "[/]"


def epoch_string(epoch: int):
    return f"\n[bold bright_white]Epoch {epoch}:[/bold bright_white] {timestamp_string()}"


def summary_string(log_line_len):
    return f"[bold bright_white]Summary:[/bold bright_white] {' ' * (log_line_len - 18)}\n"


class Tracker:
    def __init__(
        self,
        print_every: Union[int, float, None] = 1.0,
        cpu_util_window: int = 25,
        debug_epoch_break_steps: Optional[int] = float("inf"),
    ) -> None:
        """Tracks metrics, prints to console and logs to wandb.

        Example using `.epochs()` and `.steps()`:
        ```
        for epoch in tracker.epochs(num_epochs):
            for (x, x_sl), metadata in tracker.steps(train_dataloader):
                ...
                tracker.update(metrics)

            for (x, x_sl), metadata in tracker.steps(valid_dataloader):
                ...
                tracker.update(metrics)

            tracker.log()
        ```

        Example without `.epochs()` and `.steps()`
        ```
        for epoch in range(num_epochs):
            tracker.set("train", max_steps=len(train_dataloader))
            for (x, x_sl), metadata in train_dataloader:
                ...
                tracker.increment_step()
                tracker.update(metrics)
                tracker.print()
            tracker.unset()

            tracker.set("test", max_steps=len(valid_loader))
            for (x, x_sl), metadata in valid_dataloader:
                ...
                tracker.increment_step()
                tracker.update(metrics)
                tracker.print()
            tracker.unset()

            tracker.log()
            tracker.reset()
        ```

        Args:
            min_indent (int): Minimum indent for dataset name. Defaults to 40.
            print_every (Union[int, float]): Time between prints measured in steps (if int) or seconds (if float).
                                             Defaults to 1.0 (seconds).
            debug_epoch_break_steps (int): Optional number of steps at which to break all epochs for debugging purposes.
        """

        self.print_every = print_every
        self.cpu_util_window = cpu_util_window
        self.debug_epoch_break_steps = debug_epoch_break_steps

        self.terminal = Terminal()

        # dynamic variables
        self.max_source_str_len = 0
        self.max_progress_str_len = 0

        # continously updated attributes
        self.printed_last = 0
        self.log_line_len = 0
        self.cpu_utils = collections.deque(maxlen=cpu_util_window)
        self.iowait = "-"
        self.source = None
        self.start_time = defaultdict(lambda: None)
        self.end_time = defaultdict(lambda: None)
        self.epoch = 0
        self.step_within_epoch = defaultdict(lambda: 0)
        self.step_total = defaultdict(lambda: 0)
        self.max_steps = defaultdict(lambda: 0)

        self.metrics = defaultdict(dict)  # dict(source=dict(metric.name=metric))
        self.accumulated_metrics = defaultdict(lambda: defaultdict(list))  # dict(source=dict(metric.name=list(metric)))

    @property
    def values(self) -> Dict[str, Dict[str, float]]:
        """Values of metrics as nested dict"""
        return {
            source: {metric.name: metric.value for metric in self.metrics[source].values()}
            for source in self.metrics.keys()
        }

    @property
    def accumulated_values(self) -> Dict[str, Dict[str, List[float]]]:
        """Accumulated values of metrics as nested dict"""
        return {
            source: {
                metrics[0].name: [metric.value for metric in metrics]
                for metrics in self.accumulated_metrics[source].values()
            }
            for source in self.accumulated_metrics.keys()
        }

    @property
    def best_metrics(self) -> Dict[str, Dict[str, Metric]]:
        """Best metrics according to `metric.get_best` as nested dict"""
        best = dict()
        for source in self.accumulated_metrics.keys():
            best[source] = dict()
            for name, acc_metrics in self.accumulated_metrics[source].items():
                metric = acc_metrics[0].get_best(acc_metrics)
                if metric is not None:
                    best[source][f"best_{name}"] = metric
        return best

    @property
    def best_values(self) -> Dict[str, Dict[str, float]]:
        """Values of the best metrics according to `metric.get_best` as nested dict"""
        best_metrics = self.best_metrics
        return {
            source: {name: metric.value for name, metric in best_metrics[source].items()}
            for source in best_metrics.keys()
        }

    def __call__(self, loader: Union[str, DataLoader], source: Optional[str] = None, max_steps: Optional[int] = None):
        """Shortcut applicable to the standard case."""
        return self.steps(loader, source=source, max_steps=max_steps)

    def steps(
        self, iterable: Union[Iterable, DataLoader], source: Optional[str] = None, max_steps: Optional[int] = None
    ):
        if source is None and not isinstance(iterable, DataLoader):
            raise ValueError("Must provide `source` to .steps() if iterable is not a DataLoader")

        source = source if source is not None else iterable

        self.set(source, max_steps=max_steps)

        iterator = iter(iterable)

        if hasattr(iterator, "_workers"):
            workers = [psutil.Process(w.pid) for w in iterator._workers]
        else:
            workers = None

        for batch in iterator:
            yield batch
            self.increment_step()
            if self.do_print():
                self.print(workers=workers)
            if self.step_within_epoch[self.source] >= self.debug_epoch_break_steps:
                break

        self.unset()

    def increment_step(self):
        """Increment the internal step counter `self.step_within_epoch[self.source]`"""
        self.step_within_epoch[self.source] += 1

    @overload
    def epochs(self, num_epochs: int) -> Iterable[int]:
        """Yields the epoch index while printing epoch number and epoch delimiter."""
        return self.epochs(self.epoch + 1, num_epochs + 1, 1)

    @overload
    def epochs(self, start_epoch: int, num_epochs: int) -> Iterable[int]:
        return self.epochs(start_epoch, num_epochs + 1, 1)

    @overload
    def epochs(self, start_epoch: int, num_epochs: int, step: int) -> Iterable[int]:
        return self.epochs(start_epoch, num_epochs + 1, step)

    def epochs(self, *args):
        if len(args) == 1:
            start, stop, step = self.epoch + 1, args[0], 1
        elif len(args) == 2:
            start, stop, step = args[1], args[0], 1
        elif len(args) == 3:
            start, stop, step = args[1], args[0], args[2]
        else:
            raise ValueError(f"Got `args` of length {len(args)} but that must be 1, 2 or 3.")

        for epoch in range(start, stop, step):
            self.epoch = epoch
            rich.print(epoch_string(epoch), flush=True)

            yield epoch

            print("-" * (self.log_line_len or 50), flush=True)
            self.reset()

    def set(self, source: Union[str, DataLoader], max_steps: Optional[int] = None):
        """Set source name, start time and maximum number of steps if available."""
        if isinstance(source, DataLoader):
            self.source = source.dataset.source
            self.max_steps[self.source] = len(source) if max_steps is None else max_steps
        else:
            self.source = source
            self.max_steps[self.source] = max_steps

        self.start_time[self.source] = time()

    def unset(self):
        """Resets print timer and prints final line, unsets `source` and accumulates metrics."""
        self.print(end="\n")  # print line for last iteration regardless of `do_print()`

        self.end_time[self.source] = time()

        self.step_total[self.source] += self.step_within_epoch[self.source]
        for name, metric in self.metrics[self.source].items():
            self.accumulated_metrics[self.source][name].append(metric.copy())

        self.source = None
        self.printed_last = 0
        self.cpu_utils = collections.deque(maxlen=self.cpu_utils.maxlen)

    def reset(self):
        """Reset all per-source attributes"""
        self.metrics = defaultdict(dict)
        self.start_time = defaultdict(lambda: None)
        self.end_time = defaultdict(lambda: None)
        self.step_within_epoch = defaultdict(lambda: 0)
        self.max_steps = defaultdict(lambda: 0)

    def do_print(self) -> bool:
        """Print at first and last step and according to `print_every`"""
        if self.print_every is None:
            return False

        t = time()
        if isinstance(self.print_every, float):
            do_print = (t - self.printed_last) > self.print_every  # seconds
        else:
            do_print = (self.step_within_epoch[self.source] % self.print_every) == 0 or self.step_within_epoch[
                self.source
            ] == 1  # iterations

        if do_print:
            self.printed_last = t

        return do_print

    def print(self, end="\r", source: Optional[str] = None, workers: list = None):
        """Print the current progress and metric values."""
        source = self.source if source is None else source

        # progress string
        if self.max_steps[source]:
            steps_frac = f"{self.step_within_epoch[source]}/{self.max_steps[source]}"
        else:
            steps_frac = f"{self.step_within_epoch[source]}/-"

        ps = f"{steps_frac}"
        if self.start_time[source] is None:
            duration = "-"
            ms_per_step = "-"
        else:
            duration = time() - self.start_time[source]
            ms_per_step = duration / self.step_within_epoch[source] * 1000
            ms_per_step = f"{int(ms_per_step):d}ms"
            mins = int(duration // 60)
            secs = int(duration % 60)
            duration = f"{mins:d}m {secs:2d}s"

        if workers is not None:
            cpu = int(round(sum([p.cpu_percent(interval=0.0) for p in workers]), 0))  # percent since last call
            self.cpu_utils.append(cpu)
            cpu_times = [p.cpu_times() for p in workers]  # accumulated over lifetime of process
            time_usr_sys = sum([sum(ct[:2]) for ct in cpu_times]) / len(workers)
            time_iowait = sum([ct.iowait for ct in cpu_times]) / len(workers)
            self.iowait = f"{time_usr_sys:.1f}/{time_iowait:.1f}"
        if len(self.cpu_utils):
            cpu = sum(self.cpu_utils) / len(self.cpu_utils)  # avg over horizon
            cpu = f"{cpu:.0f}%"
        else:
            cpu = "-%"

        ps = f"{steps_frac} [bright_white not bold]({duration}, {ms_per_step}, {cpu} {self.iowait}s)[/]"  # +26 format

        # source string
        ss = source_string(source)

        # update dynamic string lengths
        self.max_source_str_len = max(self.max_source_str_len, len(ss))
        self.max_progress_str_len = max(self.max_progress_str_len, len(ps))

        # source progress string
        sp = f"{ss:<{self.max_source_str_len}} - {ps:<{self.max_progress_str_len}}"

        # metrics string
        sep = " [magenta]|[/] "  # " | "
        metrics = [f"{name} = {met.str_value}" for name, met in self.metrics[source].items() if met.log_to_console]
        if len(metrics) > 0:
            # maybe shorten metrics string to fit terminal
            metrics_len = [3 + len(m) for m in metrics]  # 3 is length of sep without formatting
            metrics_len[0] += 3  # add sep left of first metric
            metrics_len[-1] -= 1  # remove space right of sep right of last metric
            metrics_cumlen = list(itertools.accumulate(metrics_len))
            max_metrics_str_len = self.terminal.width - length_without_formatting(sp)
            if metrics_cumlen[-1] > max_metrics_str_len:
                # last before too long
                idx = next(i for i, v in enumerate(metrics_cumlen) if v > max_metrics_str_len - 3)
                metrics = metrics[:idx] + ["..."]
        ms = sep + sep.join(metrics)

        # final string
        s = f"{sp:<}{ms}"

        self.log_line_len = length_without_formatting(s)
        s = s + " " * 5  # add some whitespace to overwrite any lingering characters

        rich.print(s, end=end, flush=True)

    def log(self, **extra_log_data: Dict[str, Any]):
        """Log all tracked metrics to experiment tracking framework and reset `metrics`."""
        # add best and tracker metrics and any `extra_log_data`
        values = self.values
        values.update(extra_log_data)
        sources = set(values.keys()).intersection(set(self.best_values.keys()))
        for source in sources:
            values[source].update(self.best_values[source])
            values[source]["epoch_duration"] = self.end_time[source] - self.start_time[source]
            values[source]["steps"] = self.step_total[source]

        wandb.log(values)

    def update(self, metrics: List[Metric], source: Optional[str] = None, check_unique: bool = True):
        """Update all metrics tracked on `source` with the given `metrics` and add any not currently tracked"""
        source = self.source if source is None else source

        if check_unique:
            names = [metric.name for metric in metrics]
            assert len(names) == len(set(names)), "Metrics must have unique names"

        if self.start_time[source] is None:
            self.start_time[source] = time()

        for metric in metrics:
            if metric.name in self.metrics[source]:
                self.metrics[source][metric.name].update(metric)
            else:
                self.metrics[source][metric.name] = metric.copy()
