import numpy as np

from timething import align, cutter  # type: ignore


def test_pause_durations():
    alignment = align.Alignment(
        log_probs=np.zeros(1),
        trellis=np.zeros(1),
        path=np.zeros(1),
        char_segments=[],
        word_segments=[
            align.Segment("hello", 2, 8, 1.0),
            align.Segment("world", 11, 20, 1.0)
        ],
        n_frames=30,
        n_samples=100,
        ratio=0,
    )

    pauses = cutter.pause_durations(alignment)
    assert pauses == [3, 10]
