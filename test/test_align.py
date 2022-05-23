import helper
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from timething import align  # type: ignore


def test_align():
    aligner = align.Aligner("cpu", helper.StubProcessor(), helper.StubModel())
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


def test_diff():
    original, cleaned = "Ruß", "russ"
    got = list(align.diff(cleaned, original))
    want = ["- r", "+ R", "  u", "- s", "- s", "+ ß"]
    assert got == want


def test_align_cleaned_text():
    original, cleaned = "One!", "one"
    cleaned_segments = [
        helper.segment("o", 0, 1),
        helper.segment("n", 2, 3),
        helper.segment("e", 4, 5),
    ]

    want = [
        helper.segment("O", 0, 1),
        helper.segment("n", 2, 3),
        helper.segment("e!", 4, 5),
    ]

    got = align.align_clean_text(cleaned, original, cleaned_segments)
    assert want == got


def test_align_cleaned_text_number():
    original, cleaned = "0", "zero"
    cleaned_segments = [
        helper.segment("z", 0, 1),
        helper.segment("e", 2, 3),
        helper.segment("r", 4, 5),
        helper.segment("o", 6, 7),
    ]

    want = [
        helper.segment("0", 0, 7),
    ]

    got = align.align_clean_text(cleaned, original, cleaned_segments)
    assert want == got


def test_align_cleaned_text_leading_addition():
    original, cleaned = "!A", "a"
    cleaned_segments = [helper.segment("a", 0, 1)]
    got = align.align_clean_text(cleaned, original, cleaned_segments)
    want = [helper.segment("!A", 0, 1)]
    assert want == got


def test_align_cleaned_text_normalisation():
    original, cleaned = "Ruß", "russ"
    cleaned_segments = [
        helper.segment("r", 0, 1),
        helper.segment("u", 2, 3),
        helper.segment("s", 4, 5),
        helper.segment("s", 6, 7),
    ]

    want = [
        helper.segment("R", 0, 1),
        helper.segment("u", 2, 3),
        helper.segment("ß", 4, 7),
    ]

    got = align.align_clean_text(cleaned, original, cleaned_segments)
    assert want == got


def test_align_cleaned_text_commas():
    original, cleaned = "Yes, no.", "yes|no"
    cleaned_segments = [
        helper.segment("y", 0, 1),
        helper.segment("e", 2, 3),
        helper.segment("s", 4, 5),
        helper.segment("|", 6, 7),
        helper.segment("n", 8, 9),
        helper.segment("o", 10, 11),
    ]

    want = [
        helper.segment("Y", 0, 1),
        helper.segment("e", 2, 3),
        helper.segment("s", 4, 5),
        helper.segment(", ", 6, 7),
        helper.segment("n", 8, 9),
        helper.segment("o.", 10, 11),
    ]

    got = align.align_clean_text(cleaned, original, cleaned_segments)
    assert want == got


@pytest.mark.parametrize("separator", [" ", "|"])
def test_merge_words(separator):
    segs = [
        helper.segment("Y", 0, 1),
        helper.segment("e", 2, 3),
        helper.segment("s", 4, 5),
        helper.segment(separator, 6, 7),
        helper.segment("n", 8, 9),
        helper.segment("o.", 10, 11),
    ]

    want = [
        helper.segment("Yes", 0, 5),
        helper.segment("no.", 8, 11),
    ]

    got = align.merge_words(segs, separator=separator)
    assert got == want


def test_merge_words_collapsed_space():
    segs = [
        helper.segment("Y", 0, 1),
        helper.segment("e", 2, 3),
        helper.segment("s", 4, 5),
        helper.segment(", ", 6, 7),
        helper.segment("n", 8, 9),
        helper.segment("o.", 10, 11),
    ]

    want = [
        helper.segment("Yes,", 0, 5),
        helper.segment("no.", 8, 11),
    ]

    got = align.merge_words(segs, separator=" ")
    assert got == want


@given(out_text=st.text())
def test_align_cleaned_text_recovers_original(out_text):
    in_text = helper.cleaner_en(out_text)
    if in_text:
        in_segs = helper.segments(list(in_text))
        out_segs = align.align_clean_text(in_text, out_text, in_segs)
        assert "".join([s.label for s in out_segs]) == out_text
