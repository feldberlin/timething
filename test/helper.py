# Test Helpers
#
#

import contextlib
import shutil
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from timething import align, text  # type: ignore

# fixtures directory
fixtures = Path("fixtures")

# model vocab example, as for english
vocab_en = list("|'-abcdefghijklmnopqrstuvwxyz")

# english cleaner
cleaner_en = text.TextCleaner("en", vocab_en)


@contextlib.contextmanager
def tempdir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)


def alignment(
    words: typing.List[align.Segment] = [],
    chars: typing.List[align.Segment] = [],
    words_cleaned: typing.List[align.Segment] = [],
    chars_cleaned: typing.List[align.Segment] = [],
    n_model_frames: int = 30,
    n_audio_samples: int = 100,
    sampling_rate: int = 16000,
    partition_score: float = 1.0,
    alignment_score: float = 1.0,
    alignment_probability: float = 1.0,
    recognised: str = "",
    id: str = "test-alignment",
):

    return align.Alignment(
        id,
        scores=np.array([]),
        recognised=recognised,
        path=[],
        chars_cleaned=chars_cleaned or [],
        chars=chars or [],
        words_cleaned=words_cleaned or [],
        words=words or [],
        n_model_frames=n_model_frames,
        n_audio_samples=n_audio_samples,
        sampling_rate=sampling_rate,
        partition_score=partition_score,
        alignment_score=alignment_score,
        alignment_probability=alignment_probability,
    )


def segment(
    label: str,
    start: int,
    end: int,
    score: float = 1.0,
    geometric_score: float = 1.0,
):
    return align.Segment(label, start, end, score, geometric_score)


def segments(labels):
    return [segment(label, i, i + 1, 1.0) for i, label in enumerate(labels)]


def segments_eq(a, b):
    return (
        a.label == b.label
        and a.start == b.start
        and a.end == b.end
        and abs(a.score - b.score) < 10e-10
        and abs(a.geometric_score - b.geometric_score) < 10e-10
    )


def segment_lists_eq(a_list, b_list):
    return all(segments_eq(a, b) for (a, b) in zip(a_list, b_list))


@dataclass
class ProcessedBatch:
    input_values: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device):
        return self


@dataclass
class ModelOutput:
    logits: torch.Tensor


class StubModel:
    def __call__(self, batch, attention_mask):
        return ModelOutput(batch)


class StubProcessor:
    def __call__(
        self, batch, sampling_rate: int, return_tensors: str, padding: bool
    ):
        return ProcessedBatch(batch, torch.zeros_like(batch))
