import importlib.resources as pkg_resources
import json
from pathlib import Path

import numpy as np
import torchaudio  # type: ignore
import yaml  # type: ignore

import timething
from timething import align  # type: ignore

# yaml file containing all of the models
MODELS_YAML = "models.yaml"


def load_config(model: str) -> align.Config:
    """
    Load config object for the given model key
    """

    text = pkg_resources.read_text(timething, MODELS_YAML)
    cfg = yaml.safe_load(text)
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


def write_alignment(output_path: Path, id: str, alignment: align.Alignment):
    """
    Write a custom json alignments file for a given aligned recording.
    """

    def rescale(n_model_frames: int) -> float:
        return alignment.model_frames_to_seconds(n_model_frames)

    def alignments(segments):
        return [
            {
                "label": segment.label,
                "start": rescale(segment.start),
                "end": rescale(segment.end),
                "score": segment.score,
            }
            for segment in segments
        ]

    # combine the metadata
    meta = {
        "n_model_frames": alignment.n_model_frames,
        "n_audio_samples": alignment.n_audio_samples,
        "sampling_rate": alignment.sampling_rate,
        "chars": alignments(alignment.chars),
        "chars_cleaned": alignments(alignment.chars_cleaned),
        "words": alignments(alignment.words),
        "words_cleaned": alignments(alignment.words_cleaned),
    }

    # write any path components, e.g. for id 'audio/one.mp3.json'
    filename = alignment_filename(output_path, id)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # write the file
    with open(filename, "w", encoding="utf8") as f:
        f.write(json.dumps(meta, indent=4, ensure_ascii=False))


def read_alignment(alignments_dir: Path, alignment_id: str) -> align.Alignment:
    """
    Read Aligments json file.
    """

    with open(alignment_filename(alignments_dir, alignment_id), "r") as f:
        alignment_dict = json.load(f)

    alignment = align.Alignment(
        np.array([]),  # log probs
        np.array([]),  # trellis
        np.array([]),  # backtracking path
        [],  # char segments
        [],  # original char segments
        [],  # word segments
        [],  # original word segments
        alignment_dict["n_model_frames"],
        alignment_dict["n_audio_samples"],
        alignment_dict["sampling_rate"],
    )

    def rescale(n_seconds: int) -> int:
        return alignment.seconds_to_model_frames(n_seconds)

    def dict_to_segment(d: dict) -> align.Segment:
        return align.Segment(
            start=rescale(d["start"]),
            end=rescale(d["end"]),
            label=d["label"],
            score=d["score"],
        )

    alignment.chars_cleaned = [
        dict_to_segment(d) for d in alignment_dict["chars_cleaned"]
    ]

    alignment.chars = [dict_to_segment(d) for d in alignment_dict["chars"]]

    alignment.words_cleaned = [
        dict_to_segment(d) for d in alignment_dict["words_cleaned"]
    ]

    alignment.words = [dict_to_segment(d) for d in alignment_dict["words"]]

    return alignment


def alignment_filename(path, id):
    """
    From audio/one.mp3 to audio/one.mp3.json
    """

    filename = path / id
    return filename.parent / (filename.name + ".json")
