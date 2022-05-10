# Test Helpers
#
#

import typing

import numpy as np

from timething import align  # type: ignore


def alignment(n_model_frames: int, word_segments: typing.List[align.Segment]):
    return align.Alignment(
        log_probs=np.zeros(1),
        trellis=np.zeros(1),
        path=np.zeros(1),
        char_segments=[],
        word_segments=word_segments,
        n_model_frames=n_model_frames,
        n_audio_samples=100,
        sampling_rate=16000,
    )
