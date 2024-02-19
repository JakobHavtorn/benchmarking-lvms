import argparse
import datetime
import itertools
import re
import subprocess
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from random import shuffle
from typing import Union

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


def make_command_dry(cmd: str):
    return f"echo {cmd}; sleep 5; echo 'Finished {cmd}'"


def add_cuda_device(cmd: str, device_cycler: itertools.cycle):
    if "CUDA_VISIBLE_DEVICES" not in cmd:
        if "env " in cmd:
            cmd = cmd.replace("env ", "")
        cmd = f"env CUDA_VISIBLE_DEVICES={next(device_cycler).index} {cmd}"
    return cmd


def remove_repeated_whitespace(s: str):
    return re.sub(r"\s+", " ", s)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_file", type=str, help="plain text file with commands to run")
    parser.add_argument("--use_single_process", action="store_true", help="run commands in single process (using multithreading instead of multiprocessing)")
    parser.add_argument("--num_parallel", type=int, default=torch.cuda.device_count(), help="max number of parallel runs")
    parser.add_argument("--num_repeats", type=int, default=1, help="number of times to repeat each command")
    parser.add_argument("--assign_devices", type=str2bool, default=False, help="assign devices via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--shuffle_file_order", action="store_true", help="shuffle order of the commands in file")
    parser.add_argument("--dryrun", action="store_true", help="make the run dry")

    args = parser.parse_args()
    return args


def run(args, executor: Union[ThreadPoolExecutor, ProcessPoolExecutor]):
    # read commands
    with open(args.run_file, "r") as f:
        lines = f.read().strip().split("\n")

    # filter out non-commands
    commands = list(filter(is_python_command, lines))
    commands = list(map(remove_repeated_whitespace, commands))
    
    rich.print(f"Read commands:")
    rich.print(commands)

    if args.dryrun:
        commands = list(map(make_command_dry, commands))
        rich.print(f"[red]Dry run[/] - commands will not be executed.")

    # create repeats
    commands = [cmd for cmd in commands for _ in range(args.num_repeats)]

    # shuffling
    if args.shuffle_file_order:
        shuffle(commands)

    # add devices
    if args.assign_devices:
        devices = get_free_devices(n_devices=torch.cuda.device_count())
        device_cycler = itertools.cycle(devices)
        commands = list(map(partial(add_cuda_device, device_cycler=device_cycler), commands))

    rich.print(f"Commands to execute:")
    rich.print(commands)

    # Run at most `args.num_parallel` commands at a time
    futures = []
    n_failures = 0
    for command in tqdm(commands, desc=f"Running commands ({args.num_parallel} parallel)"):
        # submit command
        future = executor.submit(subprocess.run, command, check=True, shell=True, capture_output=True)
        future.command = command
        futures.append(future)
        rich.print(f"\n{get_timestamp()} Submitted {command}")
        time.sleep(10 if not args.dryrun else 0.5)

        # wait for a free slot
        while len(futures) >= args.num_parallel:
            rich.print(f"{get_timestamp()} Currently running {len(futures)} commands. Wating for one to finish.")
            future = next(as_completed(futures))

            if future.result().returncode == 0:
                rich.print(f"{get_timestamp()} {SUCCESS} {future.command}")
            else:
                n_failures += 1
                rich.print(f"{get_timestamp()} {FAILURE} {future.command}")

            futures.remove(future)

    # wait for the rest
    for future in as_completed(futures):
        if future.result().returncode == 0:
            rich.print(f"{get_timestamp()} {SUCCESS} {future.command}")
        else:
            n_failures += 1
            rich.print(f"{get_timestamp()} {FAILURE} {future.command}")
            
    rich.print(f"{get_timestamp()} Finished execution of {len(commands)} commands..")
    rich.print(f"{SUCCESS} - {len(commands) - n_failures} sucesses.")
    if n_failures:
        rich.print(f"{FAILURE} - {n_failures} failures.")


if __name__ == "__main__":
    args = parse_args()

    try:
        executor = ThreadPoolExecutor if args.use_single_process else ProcessPoolExecutor
        executor = executor(max_workers=args.num_parallel)
        run(args, executor)
    except Exception as e:
        rich.print(f"{FAILURE} - {e}")
        executor.shutdown()
        exit(1)
    finally:
        executor.shutdown()
        exit(0)
        