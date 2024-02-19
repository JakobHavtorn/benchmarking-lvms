from collections.abc import Iterable
from io import StringIO
from typing import Optional, Union, List, Any

import logging
import os
import re
import subprocess
from attr import mutable

import pandas as pd

import torch


LOGGER = logging.getLogger(__file__)


def to_device_recursive(x: Union[Any, List[Any]], device: torch.device):
    """Recursively move tensors and modules to a device ignoring non-tensor objects"""
    if x is None:
        return None

    if isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module):
        return x.to(device)

    if isinstance(x, Iterable):
        return [to_device_recursive(element, device) for element in x]

    return x


def get_global_ids_of_visible_devices():
    """Return the global indices of the visible devices.

    If `CUDA_VISIBLE_DEVICES` is not set, returns all devices.
    If it is set to the empty string, return no devices.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return list(range(torch.cuda.device_count()))

    if os.environ["CUDA_VISIBLE_DEVICES"] == "":
        return []

    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    visible_devices = re.split(r";|,| ", visible_devices)
    visible_devices = sorted([int(idx) for idx in visible_devices])
    return visible_devices


def get_device_memory_usage(do_print: bool = False) -> pd.DataFrame:
    """Return the free and used memory per GPU device on the node"""
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

    device_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")), names=["memory.used", "memory.free"], skiprows=1)

    device_df.rename(columns={"memory.used": "used", "memory.free": "free"}, inplace=True)
    device_df["free"] = device_df["free"].map(lambda x: int(x.rstrip(" [MiB]")))
    device_df["used"] = device_df["used"].map(lambda x: int(x.rstrip(" [MiB]")))

    if do_print:
        LOGGER.info("GPU usage (MiB):\n{}".format(device_df))
    return device_df


def get_free_devices(
    n_devices: Optional[int] = None, require_unused: bool = True, fallback_to_cpu: bool = True, verbose: int = 1
) -> List[torch.device]:
    """Return one or more available/visible (and unused) devices giving preference to those with most free memory"""
    device_df = get_device_memory_usage(do_print=(verbose > 1))  # All devices

    global_ids_of_visible_devices = get_global_ids_of_visible_devices()  # Visible devices

    device_df = device_df[device_df.index.isin(global_ids_of_visible_devices)]  # Only visible devices

    if require_unused:
        if require_unused == True:
            require_unused = 5  # 50 MB as default in case of boolean
        device_df = device_df[device_df.used < require_unused]

    device_df = device_df.sort_values(by="free", ascending=False)

    n_devices = n_devices or len(device_df)  # Default to all free devices

    global_device_ids = device_df.iloc[:n_devices].index.to_list()
    local_device_idx = [global_ids_of_visible_devices.index(device_id) for device_id in global_device_ids]
    devices = [torch.device(idx) for idx in local_device_idx]

    if len(devices) == n_devices:
        if verbose > 0:
            LOGGER.info(f"Found free global device(s): {global_device_ids}")
        return devices

    if fallback_to_cpu:
        msg = f"Found {len(devices)} (free) GPUs but required {n_devices}. Falling back to CPU."
        LOGGER.warning(msg)
        return torch.device("cpu")

    msg = (
        f"Found {len(devices)} (free) GPUs but required {n_devices}. "
        "If you want to fall back to CPU, set 'fallback_to_cpu=True'"
    )
    LOGGER.error(msg)
    raise RuntimeError(msg)


def get_free_device(require_unused: bool = True, fallback_to_cpu: bool = True) -> torch.device:
    """Return a single free device"""
    return get_free_devices(n_devices=1, require_unused=require_unused, fallback_to_cpu=fallback_to_cpu)[0]


def get_device(idx: Optional[int] = None):
    """Return the device to run on (cpu or cuda).

    If idx is specified, return the GPU corresponding to that index in the local scope.
    """
    if idx is None:
        return get_free_device()

    return torch.device(f"cuda:{idx}")
