import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from timething import align  # type: ignore


class Job:
    """
    An alignment job
    """

    def __init__(
        self,
        cfg: align.Config,
        loader: DataLoader,
        device: torch.device,
        output_path: Path,
    ):
        self.cfg = cfg
        self.loader = loader
        self.device = device
        self.output_path = output_path
        self.aligner = align.Aligner.build(device, cfg)

    def run(self):
        total = len(self.loader)
        for i, (xs, ys, ids) in tqdm(enumerate(self.loader), total=total):
            alignments = self.aligner.align((xs, ys))

            # write the alignments
            for i, id in enumerate(ids):
                write(self.output_path, id, alignments[i])


def write(output_path, id, alignment):
    """
    Write a custom json alignments file for a given aligned recording.
    """

    def rescale(n_frames: int) -> float:
        return n_frames * alignment.ratio

    # character aligments
    char_alignments = []
    for segment in alignment.char_segments:
        char_alignments.append(
            {
                "start": rescale(segment.start),
                "end": rescale(segment.end),
                "label": segment.label,
                "score": segment.score,
            }
        )

    # character aligments
    word_alignments = []
    for segment in alignment.word_segments:
        word_alignments.append(
            {
                "start": rescale(segment.start),
                "end": rescale(segment.end),
                "label": segment.label,
                "score": segment.score,
            }
        )

    # combine the metadata
    meta = {
        "char_alignments": char_alignments,
        "word_alignments": word_alignments,
    }

    # write the file
    filename = (output_path / id).with_suffix(".json")
    with open(filename, "w") as f:
        f.write(json.dumps(meta, indent=4, sort_keys=True))
