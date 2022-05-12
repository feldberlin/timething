# Test Helpers
#
#

import contextlib
import shutil
import tempfile
import typing
from pathlib import Path

from timething import align  # type: ignore


@contextlib.contextmanager
def tempdir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)


def alignment(
    word_segments: typing.List[align.Segment] = None,
    char_segments: typing.List[align.Segment] = None,
    n_model_frames: int = 30,
    n_audio_samples: int = 100,
    sampling_rate: int = 16000,
):

    return align.Alignment(
        log_probs=None,
        trellis=None,
        path=None,
        char_segments=char_segments or [],
        word_segments=word_segments or [],
        n_model_frames=n_model_frames,
        n_audio_samples=n_audio_samples,
        sampling_rate=sampling_rate,
    )
