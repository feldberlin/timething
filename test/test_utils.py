import helper

from timething import align, utils  # type: ignore


def test_read_write_alignment_roundtrip():
    with helper.tempdir() as tmp:
        alignment = helper.alignment(
            n_model_frames=30,
            word_segments=[
                align.Segment("hello", 2, 8, 1.0),
                align.Segment("world", 11, 20, 1.0),
            ],
        )

        alignment_id = "a01"
        utils.write_alignment(tmp, alignment_id, alignment)

        # read it back in
        got = utils.read_alignment(tmp, alignment_id)
        want = helper.alignment(
            word_segments=alignment.word_segments,
            n_model_frames=alignment.n_model_frames,
            n_audio_samples=alignment.n_audio_samples,
            sampling_rate=alignment.sampling_rate,
        )

        assert got.char_segments == want.char_segments
        assert got.word_segments == want.word_segments
        assert got.n_model_frames == want.n_model_frames
        assert got.n_audio_samples == want.n_audio_samples
        assert got.sampling_rate == want.sampling_rate
