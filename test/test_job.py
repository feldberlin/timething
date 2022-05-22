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
        assert len(ds) == 2

        print("setting up alignment job...")
        j = job.Job(cfg, ds, tmp, batch_size=1, n_workers=1, gpu=False)

        # construct the generic model text cleaner
        ds.clean_text_fn = text.TextCleaner(cfg.language, j.aligner.vocab())

        # align
        j.run()

        one = utils.read_alignment(tmp, "audio/one.mp3")
        assert len(one.words_cleaned) == 1
        assert one.words_cleaned[0].label == "one"
        assert one.words_cleaned[0].score > 0.9

        two = utils.read_alignment(tmp, "audio/two.mp3")
        assert len(two.words_cleaned) == 1
        assert two.words_cleaned[0].label == "two"
        assert two.words_cleaned[0].score > 0.9
