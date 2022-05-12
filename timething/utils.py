import json
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
    }

    # write the file
    filename = (output_path / id).with_suffix(".json")
    with open(filename, "w", encoding="utf8") as f:
        f.write(json.dumps(meta, indent=4, sort_keys=True, ensure_ascii=False))


def read_alignment(alignments_dir: Path, alignment_id: str) -> align.Alignment:
    """
    Read Aligments json file. Only contains the segmentation
    """

    with open((alignments_dir / alignment_id).with_suffix(".json"), "r") as f:
        alignment_dict = json.load(f)

    char_segments = [
        dict_to_segment(d) for d in alignment_dict["char_alignments"]
    ]

    word_segments = [
        dict_to_segment(d) for d in alignment_dict["word_alignments"]
    ]

    return align.Alignment(
        None, None, None, char_segments, word_segments, None, None, None
    )


def dict_to_segment(d: dict) -> align.Segment:
    "Convert dict to Segment"
    return align.Segment(
        start=d["start"], end=d["end"], label=d["label"], score=d["score"]
    )
