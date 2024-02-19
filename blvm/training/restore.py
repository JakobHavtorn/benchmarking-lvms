import os

from typing import Optional

import rich
import torch
import wandb

from blvm.evaluation.tracker import Tracker, timestamp_string
from blvm.models.base_model import load_model, BaseModel


CHEKPOINT_STR = "checkpoint.pt"


def save_run(
    directory: str,
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    tracker: Tracker = None,
):
    """Save a checkpoint including model and checkpoint.

    Args:
        directory (str): [description]
        model (torch.nn.Module): [description]
        optimizer (torch.optim.Optimizer, optional): [description]. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): [description]. Defaults to None.
        scaler (torch.cuda.amp.GradScaler, optional): [description]. Defaults to None.
        tracker (Tracker, optional): [description]. Defaults to None.
    """
    optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
    lr_scheduler_state_dict = lr_scheduler.state_dict() if lr_scheduler is not None else None
    scaler_state_dict = scaler.state_dict() if scaler is not None else None
    checkpoint = dict(
        epoch=tracker.epoch,
        optimizer_state_dict=optimizer_state_dict,
        lr_scheduler_state_dict=lr_scheduler_state_dict,
        scaler_state_dict=scaler_state_dict,
    )
    model.save(directory)
    torch.save(checkpoint, os.path.join(directory, CHEKPOINT_STR))
    rich.print(f"Saved checkpoint at {directory} at {timestamp_string()}")


def load_run(
    directory: str,
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    device: Optional[torch.device] = None,
    raise_errors: bool = True,
):
    """Load a run including model and checkpoint file and optionally optimizer and scaler e.g. for resuming."""
    print(f"Loading run files from: {directory}")

    if isinstance(model, BaseModel):
        device = model.device
        model = model.load(directory, device=device)
    else:
        device = device or torch.device("cpu")
        model = load_model(directory, device=device)
    print("Loaded model state dict.")

    checkpoint_path = os.path.join(directory, CHEKPOINT_STR)
    try:
        checkpoint = torch.load(checkpoint_path)
        print(f"Loaded checkpoint file.")
    except FileNotFoundError as exc:
        checkpoint = dict()
        print(f"Failed to load checkpoint file with error: {exc}")
        if raise_errors:
            raise exc

    output = []
    iterator = [
        (optimizer, "optimizer_state_dict"),
        (lr_scheduler, "lr_scheduler_state_dict"),
        (scaler, "scaler_state_dict"),
    ]
    for object, key in iterator:
        if object is None:
            continue

        try:
            object.load_state_dict(checkpoint[key])
            print(f"Loaded {key}.")
        except ValueError as exc:
            print(f"Failed to load {key} with error: {exc}")
            if raise_errors:
                raise exc

        output.append(object)

    output = [model, *output, checkpoint]  # model, (optimizer), (lr_scheduler), (scaler), checkpoint
    return tuple(output)
