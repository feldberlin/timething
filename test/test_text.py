from timething import text  # type: ignore


def test_cleaner():
    cleaner_fn = text.TextCleaner("de", list("abcdefghijklsnu "))

    # remove out of vocab letters
    assert cleaner_fn("ax bc mz") == "a|bc"

    # casefold
    assert cleaner_fn("ss ß") == "ss|ss"

    # convert ints to words
    assert cleaner_fn("11 abc 0") == "elf|abc|null"


def test_nums2words():

    # convert floats to words
    got = text.nums2words("11.2 oder 10.1", lang="de")
    want = "elf Komma zwei oder zehn Komma eins"
    assert want == got

    # convert floats to words
    got = text.nums2words("11.2", lang="de")
    want = "elf Komma zwei"
    assert want == got

    # convert negative to words
    got = text.nums2words("-11.2", lang="de")
    want = "minus elf Komma zwei"
    assert want == got
