from timething import dataset  # type: ignore


def test_cleaner():
    cleaner_fn = dataset.clean_text_fn(list("abcdefghi "))
    assert cleaner_fn("a1 bc 2z") == "a|bc"
    assert cleaner_fn("aa 88ee Aio") == "aa|ee|ai"
