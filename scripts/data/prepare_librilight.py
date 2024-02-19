import os
import tarfile
import shutil

from glob import glob

import torchaudio
import wget

from blvm.settings import DATA_DIRECTORY, SOURCE_DIRECTORY

subsets = {
    "train-10h": [],
    "train-1h": [],
    "train-10m-0": [],
    "train-10m-1": [],
    "train-10m-2": [],
    "train-10m-3": [],
    "train-10m-4": [],
    "train-10m-5": [],
}

librilight_data_dir = os.path.join(DATA_DIRECTORY, "librilight")
librilight_source_dir = os.path.join(SOURCE_DIRECTORY, "librilight")

assert not os.path.exists(librilight_data_dir), f"Dataset already exists in source directory `{librilight_data_dir}`"
assert not os.path.exists(librilight_source_dir), f"Dataset already exists in source directory `{librilight_source_dir}`"

os.mkdir(librilight_data_dir)
os.mkdir(librilight_source_dir)

# download the subset
download_url = f"https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
print(f"\nDownloading LibriLight finetuning split:")
print(f"Source: {download_url}")
print(f"Target: {librilight_data_dir}")
wget.download(download_url, librilight_data_dir)
print("\n\nSuccess!\n")

# unzip and remove downloaded tar file
download_filepath = os.path.join(librilight_data_dir, "librispeech_finetuning.tgz")
tar = tarfile.open(download_filepath, "r:gz")
tar.extractall(path=librilight_data_dir)
tar.close()
os.remove(download_filepath)

# Move directories inside extracted directory to `librilight_data_dir` root and delete extracted directory (now empty)
download_extracted_directory = os.path.join(librilight_data_dir, "librispeech_finetuning")
subdirs = os.listdir(download_extracted_directory)
for subdir in subdirs:
    shutil.move(os.path.join(download_extracted_directory, subdir), os.path.join(librilight_data_dir, subdir))
shutil.rmtree(download_extracted_directory)

# extract files for each subset non-overlapping subset
subset_subpaths = ["9h"] + [f"1h/{i}" for i in range(6)]
subsets_no = {}
for ss in subset_subpaths:
    pattern = os.path.join(librilight_data_dir, f"{ss}/*/*/*/*.flac")
    flac_paths = glob(pattern)
    flac_basenames = [f.replace(".flac", "") for f in flac_paths]
    subsets_no[ss] = flac_basenames

# define overlapping subsets
for i in range(5):
    subsets[f"train-10m-{i}"] += subsets_no[f"1h/{i}"]
    subsets["train-1h"] += subsets_no[f"1h/{i}"]
    subsets["train-10h"] += subsets_no[f"1h/{i}"]
subsets["train-10h"] += subsets_no["9h"]

# # split transcript files into single utterances
# for subset_name, examples in subsets.items():
#     print(f"\n\nSplitting transcript files - {subset_name}:")
#     transcript_filepaths = glob(os.path.join(f"{subset_name}", "*/*/*.trans.txt"))
#     source_file_content = []
#     for transcript_filepath in tqdm.tqdm(transcript_filepaths):
#         with open(transcript_filepath, "r") as transcript_file_buffer:
#             lines = transcript_file_buffer.readlines()

#         transcript_dir = os.path.split(transcript_filepath)[0]
#         for line in lines:
#             line = line.split()
#             basename = line[0]
#             transcript = " ".join(line[1:])
#             new_transcript_filepath = os.path.join(transcript_dir, f"{basename}.txt")
#             with open(new_transcript_filepath, "w") as new_transcript_file_buffer:
#                 new_transcript_file_buffer.write(transcript)

#         os.remove(transcript_filepath)

header = "filename,length.flac.samples"
for subset_name, examples in subsets.items():
    source_file_lines = [header]
    
    for f in examples:
        length_samples = torchaudio.info(f + ".flac").num_frames
        
        line = f + "," + f"{length_samples}"

        source_file_lines.append(line)
        
    source_file_content = "\n".join(source_file_lines)

    subset_source_path = os.path.join(librilight_source_dir, f"{subset_name}.txt")
    with open(subset_source_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file_content)

    print(f"Source files created at {subset_source_path}")
    
print("\nCompleted!")
