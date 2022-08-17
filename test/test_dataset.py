import base64

import helper

from timething import dataset, utils


def test_inference_dataset():
    cfg = utils.load_config("german")
    with (helper.fixtures / "audio" / "one.mp3").open("rb") as f:
        data = f.read()
        recording = base64.b64encode(data)

    records = [dataset.Base64Record(transcript="one", recording=recording)]
    ds = dataset.InferenceDataset(
        records, format="mp3", sample_rate=cfg.sampling_rate
    )

    assert len(ds) == 1
    assert len(ds[0].audio) > 0
    assert ds[0].transcript == "one"
    assert ds[0].audio.shape == (2, 64474)


def test_dataset_with_alignments():
    meta = helper.fixtures / "text.csv"
    alignments_path = helper.fixtures / "alignments"
    ds = dataset.SpeechDataset(meta, 16000, alignments_path)

    assert len(ds) == 3
    assert ds[0].alignment.words_cleaned[0].label == "one"
    assert ds[1].alignment.words_cleaned[0].label == "two"

    got = [s.label for s in ds[2].alignment.words_cleaned]
    want = ["born", "in", "nineteen", "sixty-nine", "in", "belgrade"]
    assert got == want
