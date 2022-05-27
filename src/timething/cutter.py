import shutil
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio

from timething import align, dataset, utils  # type: ignore


def pause_durations(alignment: align.Alignment) -> typing.List[float]:
    """
    Pause in number of frames after each word segment in the alignment
    """

    segments = alignment.words
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
    word_segments = alignment.words
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
    """One or more cuts from a single Recording. Each cut segment contains
    second units instead of model frames.
    """

    # the recording id
    id: str

    # one or more word level segments
    cuts: typing.List[align.Segment]


def dataset_pause_cuts(
    ds: dataset.SpeechDataset,
    cut_threshold_seconds: float,
    pause_threshold_model_frames: int,
):
    """Apply pause cuts to an entire dataset. Applies only to recordings that
    are longer than `cut_threshold_seconds`. Cuts are made where
    `pause_threshold_model_frames` is exceeded between words.
    """

    cuts = []
    for i in range(len(ds)):
        recording = ds[i]
        if recording.duration_seconds > cut_threshold_seconds:

            def rescale_seconds(n_frames: float) -> float:
                return recording.alignment.model_frames_to_seconds(n_frames)

            cuttings = pause_cuts(
                recording.alignment,
                pause_threshold_model_frames=pause_threshold_model_frames,
            )

            segments = []
            for cutting in cuttings:
                if rescale_seconds(cutting.length) <= cut_threshold_seconds:
                    segments.append(
                        align.Segment(
                            label=cutting.label,
                            start=rescale_seconds(cutting.start),
                            end=rescale_seconds(cutting.end),
                            score=cutting.score,
                        )
                    )

            if segments:
                cut = Cut(recording.id, segments)
                cuts.append(cut)

    return cuts


def dataset_recut(
    from_metadata: Path,
    to_metadata: Path,
    from_alignments: Path,
    cut_threshold_seconds: float,
    pause_threshold_model_frames: int,
    padding_ms: int,
):
    """Recut the input dataset `from`, and write it out as a new dataset `to`.

    Recordings which exceed cut_threshold_seconds are recut. Any resulting
    snip that is shorter than cut_threshold_seconds is retained; everything
    else is removed from the dataset.
    """

    # target dataset dir
    recut_dataset_dir = to_metadata.parent
    recut_dataset_dir.mkdir(parents=True, exist_ok=True)

    # construct the source dataset
    ds = dataset.SpeechDataset(from_metadata, alignments_path=from_alignments)

    # newly cut tracks only
    cuts = dataset_pause_cuts(
        ds, cut_threshold_seconds, pause_threshold_model_frames
    )

    # save each snip in a separate file
    texts = []
    cut_ids = set()
    for cut in cuts:
        cut_ids.add(cut.id)
        for i, snip in enumerate(cut.cuts):
            start = max(snip.start - padding_ms / 1000, 0)
            end = snip.end + padding_ms / 1000
            ys, sr = utils.load_slice(
                from_metadata.parent / cut.id, start, end,
            )

            # wrangle ids
            cut_id = Path(cut.id).stem
            cut_suffix = Path(cut.id).suffix.lstrip(".")
            cut_path = Path(cut.id).parent
            cut_file = cut_path / f"{cut_id}-{i}.{cut_suffix}"

            # files
            path = recut_dataset_dir / cut_file
            path.parent.mkdir(parents=True, exist_ok=True)

            # metadata
            texts.append((cut_file, snip.label))

            # save the file
            torchaudio.save(path, ys, sr, format=cut_suffix)

    # copy over remaining files
    for i in range(len(ds)):
        recording = ds[i]
        if recording.id in cut_ids:
            continue
        if recording.duration_seconds > cut_threshold_seconds:
            continue

        # files
        from_file = Path(from_metadata.parent / recording.id)
        to_file = Path(recut_dataset_dir / recording.id)
        to_file.parent.mkdir(parents=True, exist_ok=True)

        # metadata
        texts.append((recording.id, recording.original_transcript))

        # copy the file
        shutil.copy(from_file, to_file)

    # write out new metadata
    df = pd.DataFrame.from_records(texts)
    df.to_csv(to_metadata, sep="|", header=False, index=False)
