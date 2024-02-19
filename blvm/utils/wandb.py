import datetime
import os
import tqdm

from concurrent.futures import ThreadPoolExecutor
from typing import Union, List

import wandb

from blvm.settings import CHECKPOINT_DIR


def get_wandb_path(run_id: str, project: str = None, entity: str = None) -> str:
    """Create the wandb remote path for a run."""
    if entity is None and project is None:
        remote_path = f"{run_id}"
    elif entity is None:
        remote_path = f"{project}/{run_id}"
    else:
        remote_path = f"{entity}/{project}/{run_id}"
    return remote_path


def get_run(run_id: str, project: str = None, entity: str = None):
    """Retrieve a wandb.Run object for the given run_id, project and entity"""
    remote_path = get_wandb_path(run_id, project, entity)
    api = wandb.Api()
    run = api.run(remote_path)
    return run


def is_run_resumed():
    """Check whether a run resumed or to be resumed.

    May return False when wandb fails to resume due to non-existing run id
    """
    do_env_resume = "WANDB_RESUME" in os.environ and os.environ["WANDB_RESUME"] != "never"
    do_arg_resume = wandb.run is not None and wandb.run.resumed
    return do_env_resume or do_arg_resume


def find_run_on_disk(run, checkpoint_dir=CHECKPOINT_DIR, verbose: bool = False):
    run_dirs = [os.path.join(checkpoint_dir, run_dir) for run_dir in os.listdir(checkpoint_dir)]
    run_dir = sorted([run_dir for run_dir in run_dirs if run.id in run_dir])

    if len(run_dir) == 1:
        run_root = run_dir[0]
        if verbose:
            print(f"Found run on disk at {run_root}.")
        return run_root

    if len(run_dir) > 1:
        # TODO Return newest checkpoint directory
        raise NotImplementedError(f"More than one run found with ID {run.id}: {run_dir}")

    raise IOError(f"No runs found with ID {run.id}")


def restore_run(
    run: Union[str, wandb.apis.public.Run],
    project: str = None,
    entity: str = None,
    replace: bool = False,
    num_threads: int = 20,
    exclude: Union[str, List[str]] = "",
    verbose=True,
):
    """Restore files associated with a run.
    
    The `run` can be given as a `Run` object or as a `run_id` string.
    """
    if isinstance(run, str):
        # get API run object to get access to remote files
        # NOTE this functionality might be merged into the wandb.Run object in the future
        run = get_run(run, project, entity)

    # if not resumed, use path of run from API (should be the same of form `entity/project/run_id``)
    run_path = None if is_run_resumed() else os.path.join(*run.path)

    if is_run_resumed():
        run_root = None
        restore_directory = wandb.run.dir
        if verbose:
            print(f"Run `{run.id}`` is a resumed run located at {run.dir}")
    else:
        # if not resumed, try to find the run on disk, else, create a directory.
        try:
            run_root = find_run_on_disk(run, verbose=verbose)
        except IOError:
            time = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")
            run_root = os.path.join(CHECKPOINT_DIR, f"run-{time}-{run.id}-restored")
            if verbose:
                print(f"Run {run.id} neither resumed nor already located on disk. Will restore to {run_root}.")

        # put files in files subdir        
        run_root = os.path.join(run_root, "files")
        os.makedirs(run_root, exist_ok=True)
        restore_directory = run_root

    if verbose:
        print(f"Restoring files of run `{run.id}` from `{restore_directory}`.")

    files = run.files()
    if exclude:
        exclude = [exclude] if isinstance(exclude, str) else exclude
        excluded_files = [f for f in files if any([exc in f.name for exc in exclude])]
        files = list(set(files) - set(excluded_files))

    if verbose:
        print(f"Excluded {len(excluded_files)} files that won't be restored with filter: {exclude}.")
        print(f"Waiting while restoring {len(files)} files using {num_threads} threads.")

    if num_threads <= 1:
        for f in tqdm.tqdm(files):
            wandb.restore(f.name, run_path=run_path, root=run_root, replace=replace)
    else:
        pool = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="wandb_run_downloader")
        for f in files:
            pool.submit(wandb.restore, name=f.name, run_path=run_path, root=run_root, replace=replace)
        pool.shutdown(wait=True)

    if verbose:
        print(f"Restore completed. All files stored at {restore_directory}")
    return restore_directory
