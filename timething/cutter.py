import typing

import align
import numpy as np


def pause_cuts(alignment, cut_threshold=20):

    # word segments
    word_segments = alignment.word_segments

    # a list of pause durations between words
    pauses = pause_durations(alignment)

    # word segment indices that have a trailing pause
    word_segment_idxs = np.argwhere(pauses > cut_threshold).squeeze()

    # a list of (from, to) word segments that we want to cut on
    shifted = np.insert(word_segment_idxs, 0, -1) + 1
    segment_idxs = zip(shifted, word_segment_idxs)

    # construct the these larger segments
    segments = []
    for (i, j) in segment_idxs:
        label = " ".join([w.label for w in word_segments[i : j + 1]])
        segment = align.Segment(
            label, word_segments[i].start, word_segments[j].end, 1.0
        )
        segments.append(segment)

    return segments


def pause_durations(alignment: align.Alignment) -> typing.List[int]:
    """
    Pause in number of frames after each word segment in the alignment
    """

    segments = alignment.word_segments
    pauses = [y.start - x.end for (x, y) in zip(segments[:-1], segments[1:])]
    pause_end = alignment.n_frames - segments[-1].end
    pauses.append(pause_end)
    return pauses
