import helper

from timething import align, cutter  # type: ignore


def test_pause_durations():
    alignment = helper.alignment(
        n_model_frames=30,
        word_segments=[
            align.Segment("hello", 2, 8, 1.0),
            align.Segment("world", 11, 20, 1.0),
        ],
    )

    pauses = cutter.pause_durations(alignment)
    assert pauses == [3, 10]


def test_no_cut():
    alignment = helper.alignment(
        n_model_frames=30,
        word_segments=[
            align.Segment("hello", 2, 8, 1.0),
            align.Segment("world", 11, 20, 1.0),
        ],
    )

    cuts = cutter.pause_cuts(alignment, cut_threshold=100)
    assert len(cuts) == 1
    assert cuts[0].label == "hello world"
    assert cuts[0].start == 2
    assert cuts[0].end == 20


def test_one_cut():
    alignment = helper.alignment(
        n_model_frames=30,
        word_segments=[
            align.Segment("hello", 2, 8, 1.0),
            align.Segment("world", 15, 30, 1.0),
        ],
    )

    cuts = cutter.pause_cuts(alignment, cut_threshold=5)
    assert len(cuts) == 2
    assert cuts[0].label == "hello"
    assert cuts[0].start == 2
    assert cuts[0].end == 8
    assert cuts[1].label == "world"
    assert cuts[1].start == 15
    assert cuts[1].end == 30
