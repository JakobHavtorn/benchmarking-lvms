"""
This script prepares the TIMIT dataset.

Since this dataset is commercial, we do not download it in this script. That must be done beforehand.

This script then unpacks the downloaded data and creates a source file including file lengths (text and audio).
"""
import copy
import os
import random
import sys

import torchaudio

from tqdm import tqdm
from glob import glob

from blvm.settings import DATA_DIRECTORY, SOURCE_DIRECTORY


SUBSETS = [("test", "test"), ("train_all", "train")]

VALIDATION_SPLIT_SEED = 0
VALIDATION_SPLIT_SIZE = 231  # 5% of train subset

data_dir = os.path.join(DATA_DIRECTORY, "timit")
source_dir = os.path.join(SOURCE_DIRECTORY, "timit")

assert os.path.exists(data_dir), "TIMIT dataset must already be downloaded."
assert not os.path.exists(source_dir), "Dataset already exists in source directory."

os.makedirs(source_dir, exist_ok=True)

audio_ext = "flac"
header = "filename,length.wav.samples,length.flac.samples,length.txt.char,length.txt.word"

subset_extentionless_filepaths = dict()
source_file_lines = dict()
for subset, subset_dir in SUBSETS:

    # run check on files
    subset_data_dir = os.path.join(data_dir, subset_dir)
    audio_filepaths = sorted(glob(os.path.join(subset_data_dir, f"**/*.{audio_ext}"), recursive=True))
    if not audio_filepaths:  # fallback to wav
        audio_ext = "wav"
        audio_filepaths = sorted(glob(os.path.join(subset_data_dir, f"**/*.{audio_ext}"), recursive=True))
    txt_filepaths = sorted(glob(os.path.join(subset_data_dir, "**/*.TXT"), recursive=True))
    assert len(audio_filepaths) == len(txt_filepaths)
    assert len(set(audio_filepaths)) == len(audio_filepaths)
    assert len(set(txt_filepaths)) == len(txt_filepaths)

    # load files and compute length
    subset_extentionless_filepaths[subset] = [fp.replace(f".{audio_ext}", "") for fp in audio_filepaths]

    source_file_lines[subset] = []
    for file_path in tqdm(subset_extentionless_filepaths[subset]):
        length_samples = torchaudio.info(file_path + f".{audio_ext}").num_frames

        with open(file_path + ".TXT", "r") as data_file_buffer:
            txt = data_file_buffer.read()
            txt = txt.split()[2:]  # Remove unknown pre annotation e.g. "'0 46797 She had your dark suit in greasy wash water all year.\n"

            length_char = len(" ".join(txt))
            length_word = len(txt)

        line = f"{file_path},{length_samples},{length_samples},{length_char},{length_word}"
        source_file_lines[subset].append(line)

    # create subset source file
    source_file_content = "\n".join([header] + source_file_lines[subset])
    source_file_path = os.path.join(source_dir, f"{subset}.txt")
    with open(source_file_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file_content)

    print(f"Saved source file at {source_file_path} of size {sys.getsizeof(source_file_content)} bytes")


# asserts for validity
assert len(subset_extentionless_filepaths["test"]) == 1680
assert len(subset_extentionless_filepaths["train_all"]) == 4620
assert len(source_file_lines["test"]) == 1680
assert len(source_file_lines["train_all"]) == 4620

# create validation and train splits from train set
# TODO Sample validation set as 5% of training set by picking 1 utterance from every other speaker.
print(f"Creating validation split from {VALIDATION_SPLIT_SIZE} samples from `train` with seed {VALIDATION_SPLIT_SEED}")
source_lines_train_all = source_file_lines["train_all"]
random.seed(VALIDATION_SPLIT_SEED)
source_lines_valid = set(random.sample(source_lines_train_all, VALIDATION_SPLIT_SIZE))
source_lines_train = set(source_lines_train_all) - set(source_lines_valid)

# validity checks
source_lines_test = set(source_file_lines["test"])
assert len(source_lines_train & source_lines_test) == 0
assert len(source_lines_valid & source_lines_test) == 0
assert len(set(source_lines_test) & set(source_lines_test)) == len(source_lines_test)
assert sum(['test' in f for f in source_lines_train]) == 0
assert sum(['test' in f for f in source_lines_valid]) == 0

# create subset source files
source_lines_train = sorted(list(source_lines_train))
source_lines_valid = sorted(list(source_lines_valid))
for subset, source_file_lines in [("train", source_lines_train), ("valid", source_lines_valid)]:
    source_file_lines = [header] + source_file_lines
    source_file_content = "\n".join(source_file_lines)
    source_file_path = os.path.join(source_dir, f"{subset}.txt")
    with open(source_file_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file_content)

    print(f"Saved source file at {source_file_path} of size {sys.getsizeof(source_file_content)} bytes")

print("\n\nTIMIT dataset succesfully processed!")
