import typing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore
import torch
import torch.nn.utils.rnn as rnn
import torchaudio  # type: ignore
from torch.utils.data import Dataset


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

    # recording sample rate
    sample_rate: int


class SpeechDataset(Dataset):
    """
    Process a folder of audio files and transcriptions
    """

    def __init__(self, metadata: Path, resample_to: int, clean_text_fn=None):
        self.resample_to = resample_to
        self.clean_text_fn = clean_text_fn
        self.records = self.csv(metadata)

    def __getitem__(self, idx):
        """
        Return a single (audio, transcript) example from the dataset
        """

        assert idx >= 0
        assert idx <= len(self)
        record = self.records[idx]

        # read in audio
        audio, sample_rate = torchaudio.load(record.file)
        if self.resample(sample_rate):
            tf = torchaudio.transforms.Resample(sample_rate, self.resample_to)
            audio = tf(audio)

        # read and process transcription
        transcript = record.transcript
        if self.clean_text_fn:
            transcript = self.clean_text_fn(transcript)

        return Recording(record.id, audio, transcript, sample_rate)

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

    xs = [el.permute(1, 0) for el in xs]
    xs = rnn.pad_sequence(xs, batch_first=True)  # type: ignore
    xs = xs.permute(0, 2, 1)  # type: ignore

    return xs, ys, ids
