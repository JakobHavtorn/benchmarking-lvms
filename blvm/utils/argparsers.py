from functools import partial
from typing import Any, Dict, Iterable

import argparse
import json

import torch

from blvm.utils.argparsing import float_or_str, int_or_str, parse_args_by_group, str2bool
from blvm.data.datasets import DATASETS
from blvm.utils.rand import get_random_seed


parser = argparse.ArgumentParser()

# General setup
setup_group = parser.add_argument_group("setup")
setup_group.add_argument("--seed", type=int, default=get_random_seed(), help="")
setup_group.add_argument("--device", type=int_or_str, default="auto", help="")
setup_group.add_argument("--use_amp", type=str2bool, default=False, help="if true, use automatic mixed precision")
setup_group.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")

# Data config
data_group = parser.add_argument_group("data")  # input_coding, random_segment_size,
data_group.add_argument("--dataset", type=str, default=None, choices=DATASETS.keys(), help="dataset to use")

# Training related
train_group = parser.add_argument_group("training")
train_group.add_argument("--epochs", type=int, default=10, help="")
train_group.add_argument("--batch_size", type=int, default=0, help="Batch size in number of examples")
train_group.add_argument("--batch_len", type=float_or_str, default=0, help="Batch size in sequence length")
train_group.add_argument("--lr", type=float, default=3e-4, help="")
train_group.add_argument("--length_sampler", type=str2bool, default=False, help="")  # TODO Replaced by batch_len
setup_group.add_argument("--save_checkpoints", type=str2bool, default=False, help="whether to store checkpoints or not")
setup_group.add_argument("--test_every", type=int, default=10, help="test every number of epochs")

# Optimizer configuration
optim_group = parser.add_argument_group("optimizer")
optim_group.add_argument("--optimizer", type=str, default=None, help="")
optim_group.add_argument("--optimizer_kwargs", type=json.loads, default=dict(), help="")
optim_group.add_argument("--max_grad_norm", type=float, default=float("inf"), help="Gradient clipping by norm max.")
optim_group.add_argument("--max_grad_value", type=float, default=float("inf"), help="Gradient clipping by value max.")

# Learning rate scheduler configuration
sched_group = parser.add_argument_group("scheduler")
optim_group.add_argument("--lr_scheduler", type=str, default="ExponentialLR", help="")
optim_group.add_argument("--lr_scheduler_kwargs", type=json.loads, default=dict(gamma=1), help="")

# Distributed data parallel configuration
distr_group = parser.add_argument_group("distributed data parallel")
distr_group.add_argument("--ddp_master_addr", default=None, type=str, help="address for the DDP master")
distr_group.add_argument("--ddp_master_port", default=None, type=str, help="port for the DDP master")
distr_group.add_argument("--nodes", "-n", default=None, type=int, help="number of nodes")
distr_group.add_argument("--gpus", "-g", default=None, type=int, help="number of gpus per node")
distr_group.add_argument("--node_rank", "-nr", default=None, type=int, help="ranking of this node within the nodes")

# Wandb configuration
# https://docs.wandb.ai/ref/python/init
wandb_group = parser.add_argument_group("wandb")
wandb_group.add_argument("--entity", type=str, default=None)
wandb_group.add_argument("--project", type=str, default=None)
wandb_group.add_argument("--id", type=str, default=None)
wandb_group.add_argument("--name", type=str, default=None)
wandb_group.add_argument("--tags", type=str, nargs="+", default=None)
wandb_group.add_argument("--group", type=str, default=None)
wandb_group.add_argument("--notes", type=str, default=None)
wandb_group.add_argument("--resume", type=str, default=None)
wandb_group.add_argument("--mode", type=str, default=None, choices=["online", "offline", "disabled"])
wandb_group.add_argument("--job_type", type=str, default=None)


# monkey patch argparser with per-group parsing function
parser.parse_args_by_group = partial(parse_args_by_group, parser=parser)


def get_optimizer(parameters: Iterable[torch.nn.Module], optimizer: str, **optimizer_kwargs: Dict[str, Any]):
    """Returns an Optimizer of class `optimizer` instantiated with `optimizer_kwargs`."""
    optimizer_class = getattr(torch.optim, optimizer)
    optimizer = optimizer_class(parameters, **optimizer_kwargs)
    return optimizer


def get_lr_scheduler(optimizer: torch.optim.Optimizer, lr_scheduler: str, **lr_scheduler_kwargs: Dict[str, Any]):
    """Returns an LRScheduler of class `lr_scheduler` instantiated with `lr_scheduler_kwargs`."""
    lr_scheduler_class = getattr(torch.optim.lr_scheduler, lr_scheduler)
    lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
    return lr_scheduler
