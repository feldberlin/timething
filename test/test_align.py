from dataclasses import dataclass

import helper
import torch

from timething import align  # type: ignore


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


def test_align():
    aligner = align.Aligner("cpu", StubProcessor(), StubModel())
    batch = torch.rand((2, 1, 20))
    logits = aligner.logp(batch)
    assert logits.equal(torch.log_softmax(batch.squeeze(1), dim=-1))


def test_alignment_time_units():

    # 2 seconds of 16k audio at 10 frames per second
    alignment = helper.alignment(
        n_model_frames=200, n_audio_samples=32000, sampling_rate=16000
    )

    assert alignment.model_frames_to_seconds(100) == 1.0
    assert alignment.model_frames_to_fraction(10) == 0.05
    assert alignment.seconds_to_model_frames(2.0) == 200

    # 0.00625 seconds of 16k audio at 4800 frames per second
    alignment = helper.alignment(
        n_model_frames=30, n_audio_samples=100, sampling_rate=16000
    )

    n_frames = 20
    seconds = alignment.model_frames_to_seconds(n_frames)
    assert alignment.seconds_to_model_frames(seconds) == n_frames
