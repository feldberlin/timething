from pathlib import Path
import importlib.resources as pkg_resources
import json

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
        "n_model_frames": alignment.n_model_frames,
        "n_audio_samples": alignment.n_audio_samples,
        "sampling_rate": alignment.sampling_rate,
    }

    # write the file
    filename = (output_path / id).with_suffix(".json")
    with open(filename, "w", encoding="utf8") as f:
        f.write(json.dumps(meta, indent=4, sort_keys=True, ensure_ascii=False))


def read_alignment(alignments_dir: Path, alignment_id: str) -> align.Alignment:
    """
    Read Aligments json file.
    """

    with open((alignments_dir / alignment_id).with_suffix(".json"), "r") as f:
        alignment_dict = json.load(f)

    alignment = align.Alignment(
        np.array([]),  # log probs
        np.array([]),  # trellis
        np.array([]),  # backtracking path
        [],  # char segments
        [],  # word segments
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

    alignment.char_segments = [
        dict_to_segment(d) for d in alignment_dict["char_alignments"]
    ]

    alignment.word_segments = [
        dict_to_segment(d) for d in alignment_dict["word_alignments"]
    ]

    return alignment
