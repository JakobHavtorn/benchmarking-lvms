import os

from blvm.settings import SOURCE_DIRECTORY


LIBRISPEECH = "librispeech"  # dataset
LIBRISPEECH_100H = "librispeech_100h"
LIBRISPEECH_TRAIN = "librispeech_train"
LIBRISPEECH_TRAIN_CLEAN_100 = "librispeech_train_clean_100"
LIBRISPEECH_TRAIN_CLEAN_360 = "librispeech_train_clean_360"
LIBRISPEECH_TRAIN_OTHER_500 = "librispeech_train_other_500"
LIBRISPEECH_DEV_CLEAN = "librispeech_dev_clean"
LIBRISPEECH_DEV_OTHER = "librispeech_dev_other"
LIBRISPEECH_TEST_CLEAN = "librispeech_test_clean"
LIBRISPEECH_TEST_OTHER = "librispeech_test_other"

LIBRILIGHT = "librilight"  # dataset
LIBRILIGHT_TRAIN_10H = "librilight_train_10h"
LIBRILIGHT_TRAIN_1H = "librilight_train_1h"
LIBRILIGHT_TRAIN_10M0 = "librilight_train_10m0"
LIBRILIGHT_TRAIN_10M1 = "librilight_train_10m1"
LIBRILIGHT_TRAIN_10M2 = "librilight_train_10m2"
LIBRILIGHT_TRAIN_10M3 = "librilight_train_10m3"
LIBRILIGHT_TRAIN_10M4 = "librilight_train_10m4"
LIBRILIGHT_TRAIN_10M5 = "librilight_train_10m5"

TIMIT = "timit"  # dataset
TIMIT_TRAIN = "timit_train"
TIMIT_TRAIN_FULL = "timit_train_full"
TIMIT_VALID = "timit_valid"
TIMIT_TEST = "timit_test"

DATAPATHS_MAPPING = {
    LIBRISPEECH_TRAIN: os.path.join(SOURCE_DIRECTORY, LIBRISPEECH, "train.txt"),
    LIBRISPEECH_TRAIN_CLEAN_100: os.path.join(SOURCE_DIRECTORY, LIBRISPEECH, "train-clean-100.txt"),
    LIBRISPEECH_TRAIN_CLEAN_360: os.path.join(SOURCE_DIRECTORY, LIBRISPEECH, "train-clean-360.txt"),
    LIBRISPEECH_TRAIN_OTHER_500: os.path.join(SOURCE_DIRECTORY, LIBRISPEECH, "train-other-500.txt"),
    LIBRISPEECH_DEV_CLEAN: os.path.join(SOURCE_DIRECTORY, LIBRISPEECH, "dev-clean.txt"),
    LIBRISPEECH_DEV_OTHER: os.path.join(SOURCE_DIRECTORY, LIBRISPEECH, "dev-other.txt"),
    LIBRISPEECH_TEST_CLEAN: os.path.join(SOURCE_DIRECTORY, LIBRISPEECH, "test-clean.txt"),
    LIBRISPEECH_TEST_OTHER: os.path.join(SOURCE_DIRECTORY, LIBRISPEECH, "test-other.txt"),
    LIBRILIGHT_TRAIN_10H: os.path.join(SOURCE_DIRECTORY, LIBRILIGHT, "train-10h.txt"),
    LIBRILIGHT_TRAIN_1H: os.path.join(SOURCE_DIRECTORY, LIBRILIGHT, "train-1h.txt"),
    LIBRILIGHT_TRAIN_10M0: os.path.join(SOURCE_DIRECTORY, LIBRILIGHT, "train-10m-0.txt"),
    LIBRILIGHT_TRAIN_10M1: os.path.join(SOURCE_DIRECTORY, LIBRILIGHT, "train-10m-1.txt"),
    LIBRILIGHT_TRAIN_10M2: os.path.join(SOURCE_DIRECTORY, LIBRILIGHT, "train-10m-2.txt"),
    LIBRILIGHT_TRAIN_10M3: os.path.join(SOURCE_DIRECTORY, LIBRILIGHT, "train-10m-3.txt"),
    LIBRILIGHT_TRAIN_10M4: os.path.join(SOURCE_DIRECTORY, LIBRILIGHT, "train-10m-4.txt"),
    LIBRILIGHT_TRAIN_10M5: os.path.join(SOURCE_DIRECTORY, LIBRILIGHT, "train-10m-5.txt"),
    TIMIT_TRAIN: os.path.join(SOURCE_DIRECTORY, TIMIT, "train.txt"),
    TIMIT_TRAIN_FULL: os.path.join(SOURCE_DIRECTORY, TIMIT, "train_full.txt"),
    TIMIT_VALID:  os.path.join(SOURCE_DIRECTORY, TIMIT, "valid.txt"),
    TIMIT_TEST: os.path.join(SOURCE_DIRECTORY, TIMIT, "test.txt"),
}
