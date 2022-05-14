import helper
import pytest

from timething import dataset, job, utils


@pytest.mark.integration
def test_job():
    cfg = utils.load_config("german")
    metadata = helper.fixtures / "text.csv"
    with helper.tempdir() as tmp:

        # set up dataset
        ds = dataset.SpeechDataset(metadata, cfg.sampling_rate)
        assert len(ds) == 2

        print("setting up alignment job")
        j = job.Job(cfg, ds, tmp, batch_size=1, n_workers=1)

        print("aligning")
        j.run()

    one = utils.read_alignment(tmp, "audio/one.mp3")
    two = utils.read_alignment(tmp, "audio/two.mp3")

    assert len(one.word_segments) == 1
    assert one.word_segments[0] == "one"
    assert len(two.word_segments) == 1
    assert two.word_segments[0] == "two"
