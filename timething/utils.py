from pathlib import Path

import torchaudio  # type: ignore
import yaml  # type: ignore

from timething import align  # type: ignore

# yaml file containing all of the models
MODELS_YAML = "timething/models.yaml"


def load_config(model: str) -> align.Config:
    """
    Load config object for the given model key
    """

    with open(MODELS_YAML, "r") as f:
        cfg = yaml.safe_load(f)
        return align.Config(
            cfg[model]["model"],
            cfg[model]["pin"],
            cfg[model]["sampling_rate"],
            cfg[model]["language"],
        )


def load_slice(filename: Path, start_seconds: float, end_seconds: float):
    """
    Load an audio slice from a seconds offset and duration using torchaudio.
    """

    info = torchaudio.info(filename)
    num_samples = torchaudio.load(filename)[0].shape[1]
    n_seconds = num_samples / info.sample_rate
    seconds_per_frame = n_seconds / info.num_frames
    start = int(start_seconds / seconds_per_frame)
    end = int(end_seconds / seconds_per_frame)
    duration = end - start
    return torchaudio.load(filename, start, duration)
