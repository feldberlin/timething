import typing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore
import torch
import torch.nn.utils.rnn as rnn
import torchaudio  # type: ignore
from torch.utils.data import Dataset

from timething import align, utils


@dataclass
class CSVRecord:
    """
    A line in the dataset metadata csv
    """

    # id in the dataset
    id: str

    # path a sigle chapter audio file
    file: Path

    # corresponding transcript
    transcript: str


@dataclass
class Recording:
    """
    A single example recording
    """

    # id in the dataset
    id: str

    # audio data
    audio: torch.Tensor

    # corresponding transcript
    transcript: str

    # transcript before cleaning
    original_transcript: str

    # the alignment for this recording, if present on disk
    alignment: typing.Optional[align.Alignment]

    # recording sample rate
    sample_rate: int

    @property
    def duration_seconds(self):
        return self.audio.shape[-1] / self.sample_rate


class SpeechDataset(Dataset):
    """
    Process a folder of audio files and transcriptions
    """

    def __init__(
        self,
        metadata: Path,
        resample_to: int,
        alignments_path: typing.Optional[Path] = None,
        clean_text_fn=None,
    ):
        self.resample_to = resample_to
        self.clean_text_fn = clean_text_fn
        self.records = self.csv(metadata)
        self.alignments_path = alignments_path

    def __getitem__(self, idx):
        """
        Return a single (audio, transcript) example from the dataset
        """

        assert idx >= 0
        assert idx <= len(self)
        record = self.records[idx]

        # read in audio
        audio, sample_rate = torchaudio.load(record.file)

        # resample, if needed
        if self.resample(sample_rate):
            tf = torchaudio.transforms.Resample(sample_rate, self.resample_to)
            audio = tf(audio)

        # squash to or retain mono
        audio = torch.mean(audio, 0, keepdim=True)

        # read and process transcription
        transcript = record.transcript
        if self.clean_text_fn:
            transcript = self.clean_text_fn(transcript)

        # read in aligments if they exist on disk
        alignment = None
        if self.alignments_path:
            path = utils.alignment_filename(self.alignments_path, record.id)
            if path.exists():
                alignment = utils.read_alignment(
                    self.alignments_path, alignment_id=record.id
                )

        return Recording(
            record.id,
            audio,
            transcript,
            record.transcript,
            alignment,
            sample_rate,
        )

    def __len__(self):
        "number of examples in this dataset"
        return len(self.records)

    def resample(self, sample_rate) -> bool:
        "should examples be resampled or not"
        return sample_rate != self.resample_to

    def csv(self, metadata: Path) -> typing.List[CSVRecord]:
        "read in the dataset csv"
        df = pd.read_csv(metadata, delimiter="|", names=("id", "transcript"))
        records = []
        for (_, row) in df.iterrows():
            file_path = metadata.parent / row.id
            records.append(CSVRecord(row.id, file_path, row.transcript))

        return records


def collate_fn(recordings: typing.List[Recording]):
    """
    Collate invididual examples into a single batch
    """

    ids = [r.id for r in recordings]
    xs = [r.audio for r in recordings]
    ys = [r.transcript for r in recordings]
    ys_original = [r.original_transcript for r in recordings]

    xs = [el.permute(1, 0) for el in xs]
    xs = rnn.pad_sequence(xs, batch_first=True)  # type: ignore
    xs = xs.permute(0, 2, 1)  # type: ignore

    return xs, ys, ys_original, ids
