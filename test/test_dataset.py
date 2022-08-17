import base64

import helper

from timething import dataset, utils


def test_inference_dataset():
    cfg = utils.load_config("german")
    with (helper.fixtures / "audio" / "one.mp3").open("rb") as f:
        one = base64.b64encode(f.read())
    records = [dataset.Base64Record(transcript="one", recording=one)]
    inference_ds = dataset.InferenceDataset(
        records, format="mp3", sample_rate=cfg.sampling_rate
    )

    # some basic checks
    assert len(inference_ds) == 1
    assert len(inference_ds[0].audio) > 0
    assert inference_ds[0].transcript == "one"
    assert inference_ds[0].audio.shape == (1, 64474)  # must be mono

    # compare shapes to speech dataset
    meta = helper.fixtures / "text.csv"
    alignments_path = helper.fixtures / "alignments"
    speech_ds = dataset.SpeechDataset(meta, 16000, alignments_path)

    assert speech_ds[0].audio.shape == inference_ds[0].audio.shape


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
