from types import SimpleNamespace

from blvm.data.datapaths import *
from blvm.settings import DATA_DIRECTORY


DATASETS = {
    TIMIT: SimpleNamespace(
        name=TIMIT,
        train=TIMIT_TRAIN,
        valid=TIMIT_VALID,
        test=TIMIT_TEST,
        valid_sets=[TIMIT_VALID],
        test_sets=[TIMIT_TEST],
        audio_ext="flac",
        audio_length="length.flac.samples",
        speaker_info=os.path.join(DATA_DIRECTORY, TIMIT, "DOC", "SPKRINFO.TXT"),
    ),
    LIBRISPEECH: SimpleNamespace(
        name=LIBRISPEECH,
        train=LIBRISPEECH_TRAIN,
        valid=LIBRISPEECH_DEV_CLEAN,
        test=LIBRISPEECH_TEST_CLEAN,
        valid2=LIBRISPEECH_DEV_OTHER,
        test2=LIBRISPEECH_TEST_OTHER,
        valid_sets=[LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_DEV_OTHER],
        test_sets=[LIBRISPEECH_TEST_CLEAN, LIBRISPEECH_TEST_OTHER],
        audio_ext="flac",
        audio_length="length.flac.samples",
    ),
    LIBRISPEECH_100H: SimpleNamespace(
        name=LIBRISPEECH_100H,
        train=LIBRISPEECH_TRAIN_CLEAN_100,
        valid=LIBRISPEECH_DEV_CLEAN,
        test=LIBRISPEECH_TEST_CLEAN,
        valid2=LIBRISPEECH_DEV_OTHER,
        test2=LIBRISPEECH_TEST_OTHER,
        valid_sets=[LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_DEV_OTHER],
        test_sets=[LIBRISPEECH_TEST_CLEAN, LIBRISPEECH_TEST_OTHER],
        audio_ext="flac",
        audio_length="length.flac.samples",
    ),
    LIBRILIGHT_TRAIN_10H: SimpleNamespace(
        name=LIBRILIGHT_TRAIN_10H,
        train=LIBRILIGHT_TRAIN_10H,
        valid=LIBRISPEECH_DEV_CLEAN,
        test=LIBRISPEECH_TEST_CLEAN,
        valid2=LIBRISPEECH_DEV_OTHER,
        test2=LIBRISPEECH_TEST_OTHER,
        valid_sets=[LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_DEV_OTHER],
        test_sets=[LIBRISPEECH_TEST_CLEAN, LIBRISPEECH_TEST_OTHER],
        audio_ext="flac",
        audio_length="length.flac.samples",
    ),
}
