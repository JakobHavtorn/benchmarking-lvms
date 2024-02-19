import argparse
import datetime
import itertools
import subprocess
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from random import shuffle

import rich
import torch

from tqdm import tqdm

from blvm.utils.device import get_free_devices
from blvm.utils.argparsing import str2bool


SUCCESS = f"[green bold]SUCCESS[/]"
FAILURE = f"[red bold]FAILURE[/]"


def get_timestamp():
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[grey bold]{t}[/]"


def is_python_command(cmd: str):
    return len(cmd) > 0 and "#" not in cmd and "python" in cmd


def add_cuda_device(cmd: str):
    if "CUDA_VISIBLE_DEVICES" not in cmd:
        if "env " in cmd:
            cmd = cmd.replace("env ", "")
        cmd = f"env CUDA_VISIBLE_DEVICES={next(device_cycler).index} {cmd}"
    return cmd


parser = argparse.ArgumentParser()

parser.add_argument("--run_file", type=str, help="plain text file with commands to run")
parser.add_argument("--use_single_process", action="store_true", help="run commands in single process (using multithreading instead of multiprocessing)")
parser.add_argument("--num_parallel", type=int, default=torch.cuda.device_count(), help="max number of parallel runs")
parser.add_argument("--num_repeats", type=int, default=1, help="number of times to repeat each command")
parser.add_argument("--assign_devices", type=str2bool, default=False, help="assign devices via CUDA_VISIBLE_DEVICES")
parser.add_argument("--shuffle_file_order", action="store_true", help="shuffle order of the commands in file")

args = parser.parse_args()

# executor
executor = ThreadPoolExecutor if args.use_single_process else ProcessPoolExecutor
executor = executor(max_workers=args.num_parallel)

# read commands
with open(args.run_file, "r") as f:
    lines = f.read().strip().split("\n")

# filter out non-commands
commands = list(filter(is_python_command, lines))
rich.print(f"Read commands:")
rich.print(commands)

# create repeats
commands = [cmd for cmd in commands for _ in range(args.num_repeats)]

# shuffling
if args.shuffle_file_order:
    shuffle(commands)

# add devices
if args.assign_devices:
    devices = get_free_devices(n_devices=torch.cuda.device_count())
    device_cycler = itertools.cycle(devices)
    commands = list(map(add_cuda_device, commands))

rich.print(f"Commands to execute:")
rich.print(commands)

futures = []
for command in tqdm(commands, desc="Submitting runs"):
    future = executor.submit(subprocess.run, command, check=True, shell=True, capture_output=True)
    future.command = command
    futures.append(future)
    if len(futures) <= args.num_parallel:
        time.sleep(10)  # sleep between submitting runs that run in parallel to allow time to allocate on GPU.

n_failures = 0
rich.print("Running commands...")
for future in as_completed(futures):
    if future.exception():
        rich.print(f"{get_timestamp()} - {FAILURE}: {future.exception()}")
        n_failures += 1
    else:
        rich.print(f"{get_timestamp()} - {SUCCESS}: {future.command}")


rich.print(f"Finished execution of {len(commands)} commands.")
rich.print(f"{SUCCESS} - {len(commands) - n_failures} sucesses.")
if n_failures:
    rich.print(f"{FAILURE} - {n_failures} failures.")

    rich.print("ERROR LOG:")
    for future in futures:
        if future.exception():
            rich.print("-" * 80)
            rich.print(f"{future.exception()}")
            rich.print(f"{future.command}")
            rich.print("-" * 80)
