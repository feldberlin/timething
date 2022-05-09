from dataclasses import dataclass

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
