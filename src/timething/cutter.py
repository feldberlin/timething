import typing
from dataclasses import dataclass

import numpy as np

from timething import align, dataset  # type: ignore


def pause_durations(alignment: align.Alignment) -> typing.List[int]:
    """
    Pause in number of frames after each word segment in the alignment
    """

    segments = alignment.words_cleaned
    pauses = [y.start - x.end for (x, y) in zip(segments[:-1], segments[1:])]
    pause_end = alignment.n_model_frames - segments[-1].end
    pauses.append(pause_end)
    return pauses


def pause_cuts(
    alignment: align.Alignment, pause_threshold_model_frames=20
) -> typing.List[align.Segment]:
    """
    Return recut Segments for a given alignment. Will cut where there is more
    than `pause_threshold_model_frames` between aligned words. Returns
    a single collapsed segment where we could not cut.
    """

    # word segments
    word_segments = alignment.words_cleaned
    n_words = len(word_segments)

    # a list of pause durations between words
    pauses = np.array(pause_durations(alignment))

    # word segment indices that have a trailing pause
    pause_segment_idxs = np.argwhere(pauses > pause_threshold_model_frames)
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


@dataclass
class Cut:

    # the alignment that has been recut
    alignment: align.Alignment

    # one or more word level segments
    cuts: typing.List[align.Segment]


def dataset_pause_cuts(
    ds: dataset.SpeechDataset,
    cut_threshold_seconds: int = 8,
    pause_threshold_model_frames: int = 20,
):
    """
    Apply pause cuts to an entire dataset. Applies only to recordings that are
    longer than `cut_threshold_seconds`.
    """

    cuts = []
    for i in range(len(ds)):
        recording = ds[i]
        if recording.duration_seconds > cut_threshold_seconds:

            # cut into smaller pieces if possible
            segments = pause_cuts(
                recording.alignment,
                pause_threshold_model_frames=pause_threshold_model_frames,
            )

            # append result
            cuts.append(Cut(recording.alignment, segments))

    return cuts
