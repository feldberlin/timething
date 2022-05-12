import json
import os

import helper

from timething import align, utils  # type: ignore


def test_write_alignment():
    with helper.tempdir() as tmp:
        alignment = helper.alignment(
            30,
            [
                align.Segment("hello", 2, 8, 1.0),
                align.Segment("world", 11, 20, 1.0),
            ],
        )

        def rescale(n_model_frames: int) -> float:
            return alignment.model_frames_to_seconds(n_model_frames)

        alignment_id = "a01"
        filename = (tmp / alignment_id).with_suffix(".json")
        utils.write_alignment(tmp, alignment_id, alignment)

        # wrote the file
        assert os.stat(filename)

        # check format
        with open(filename, "r") as f:
            alignment_dict = json.load(f)

        assert "char_alignments" in alignment_dict
        assert "word_alignments" in alignment_dict

        got = [
            utils.dict_to_segment(d) for d in alignment_dict["word_alignments"]
        ]

        want = [
            align.Segment(
                start=rescale(s.start),
                end=rescale(s.end),
                label=s.label,
                score=s.score,
            )
            for s in alignment.word_segments
        ]

        assert got == want


def test_read_alignment():
    with helper.tempdir() as tmp:
        alignment = helper.alignment(
            30,
            [
                align.Segment("hello", 2, 8, 1.0),
                align.Segment("world", 11, 20, 1.0),
            ],
        )

        def rescale(n_model_frames: int) -> float:
            return alignment.model_frames_to_seconds(n_model_frames)

        alignment_id = "a01"
        filename = (tmp / alignment_id).with_suffix(".json")
        utils.write_alignment(tmp, alignment_id, alignment)

        # read it back in
        got = utils.read_alignment(tmp, alignment_id)

        # construct what we expect to see
        word_segments = [
            align.Segment(
                start=rescale(s.start),
                end=rescale(s.end),
                label=s.label,
                score=s.score,
            )
            for s in alignment.word_segments
        ]

        assert got == align.Alignment(
            None, None, None, [], word_segments, None, None, None
        )
