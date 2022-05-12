import typing

import numpy as np

from timething import align  # type: ignore


def pause_cuts(
    alignment: align.Alignment, cut_threshold=20
) -> typing.List[align.Segment]:

    # word segments
    word_segments = alignment.word_segments
    n_words = len(word_segments)

    # a list of pause durations between words
    pauses = np.array(pause_durations(alignment))

    # word segment indices that have a trailing pause
    pause_segment_idxs = np.argwhere(pauses > cut_threshold)
    pause_segment_idxs = pause_segment_idxs.squeeze().reshape(-1)

    # a list of (from, to) word segments that we want to cut on
    bounds = []
    if len(pause_segment_idxs) == 0:
        bounds.append((0, n_words - 1))
    else:
        from_word = 0
        for to_word in pause_segment_idxs:
            bounds.append((from_word, to_word))
            from_word = to_word + 1

        # catch a possibly trailing segment
        if pause_segment_idxs[-1] != n_words - 1:
            bounds.append((from_word, n_words - 1))

    # construct the larger cut segments
    cuts = []
    for (i, j) in bounds:
        label = " ".join([w.label for w in word_segments[i : j + 1]])
        segment = align.Segment(
            label, word_segments[i].start, word_segments[j].end, 1.0
        )
        cuts.append(segment)

    return cuts


def pause_durations(alignment: align.Alignment) -> typing.List[int]:
    """
    Pause in number of frames after each word segment in the alignment
    """

    segments = alignment.word_segments
    pauses = [y.start - x.end for (x, y) in zip(segments[:-1], segments[1:])]
    pause_end = alignment.n_model_frames - segments[-1].end
    pauses.append(pause_end)
    return pauses
