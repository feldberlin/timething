# Test Helpers
#
#

import contextlib
import shutil
import tempfile
import typing
from pathlib import Path

import numpy as np

from timething import align  # type: ignore

# fixtures directory
fixtures = Path("fixtures")


@contextlib.contextmanager
def tempdir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)


def alignment(
    word_segments: typing.List[align.Segment] = [],
    char_segments: typing.List[align.Segment] = [],
    n_model_frames: int = 30,
    n_audio_samples: int = 100,
    sampling_rate: int = 16000,
):

    return align.Alignment(
        log_probs=np.array([]),
        trellis=np.array([]),
        path=np.array([]),
        char_segments=char_segments or [],
        word_segments=word_segments or [],
        n_model_frames=n_model_frames,
        n_audio_samples=n_audio_samples,
        sampling_rate=sampling_rate,
    )
