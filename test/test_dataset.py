import helper

from timething import dataset


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
