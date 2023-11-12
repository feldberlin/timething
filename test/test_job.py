import helper
import pytest

from timething import dataset, job, text, utils


@pytest.mark.integration
def test_job():
    cfg = utils.load_config("english")
    metadata = helper.fixtures / "text.csv"
    with helper.tempdir() as tmp:
        # set up dataset
        ds = dataset.SpeechDataset(metadata, cfg.sampling_rate)

        print("setting up alignment job...")
        j = job.Job(
            cfg, ds, batch_size=1, n_workers=1, gpu=False, output_path=tmp
        )

        # construct the generic model text cleaner
        ds.clean_text_fn = text.TextCleaner(cfg.language, j.aligner.vocab)

        # align
        j.run()

        one = utils.read_alignment(tmp, "audio/one.mp3")
        assert len(one.words) == 1
        assert len(one.words_cleaned) == 1
        assert one.words[0].label == "One!"
        assert one.words_cleaned[0].label == "one"
        assert one.words_cleaned[0].score > 0.9

        two = utils.read_alignment(tmp, "audio/two.mp3")
        assert len(two.words) == 1
        assert len(two.words_cleaned) == 1
        assert two.words[0].label == "Two?"
        assert two.words_cleaned[0].label == "two"
        assert two.words_cleaned[0].score > 0.9

        born = utils.read_alignment(tmp, "audio/born.mp3")
        assert born.words_cleaned[0].score > 0.8

        # cleaned
        assert len(born.words_cleaned) == 6
        assert [w.label for w in born.words_cleaned] == [
            "born",
            "in",
            "nineteen",
            "sixty-nine",
            "in",
            "belgrade",
        ]

        # original
        assert len(born.words) == 5
        assert [w.label for w in born.words] == [
            "Born",
            "in",
            "1969",
            "in",
            "Belgrade.",
        ]


@pytest.mark.manual
def test_long_audio_job():
    cfg = utils.load_config("english")
    keanu_audio = helper.fixtures / "audio" / "keanu.mp3"
    with open(helper.fixtures / "keanu.cleaned.txt", "r") as f:
        keanu_transcript = f.read()

    # set up non overlapping dataset
    ms_per_sample = 500
    hopsize_ms = ms_per_sample
    ds = dataset.WindowedTrackDataset(
        keanu_audio,
        "mp3",
        keanu_transcript,
        ms_per_sample,
        hopsize_ms,
        16000,
    )

    print("setting up alignment job...")
    j = job.LongTrackJob(
        cfg,
        ds,
        batch_size=1,
        n_workers=1,
    )

    # align
    results = j.run()
    print(results)
    assert len(results) == 1
