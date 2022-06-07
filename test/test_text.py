import numpy as np
import torch

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


def test_nums2words_year():
    got = text.nums2words("geboren 1968 in Belgrad", lang="de")
    want = "geboren neunzehnhundertachtundsechzig in Belgrad"
    assert want == got


def test_ctc_collapse():
    got = text.ctc_collapse("a <b> a b b <b> c|d <b>".split(" "), blank="<b>")
    want = "aabc d"
    assert got == want


def test_k_shingle():
    got = text.k_shingle("abcdefg", k=3)
    want = set(["abc", "bcd", "cde", "def", "efg"])
    assert got == want


def test_jaquard():
    got = text.jaquard(set(list("abc")), set(list("bcd")))
    assert got == 0.5


def test_similarity():
    got = text.similarity("abcde", "abcx", k=3)
    assert got == 0.25
    got = text.similarity("one", "one", k=3)
    assert got == 1.0


def test_decode_best(tokens=list("abcde"), n_frames=12):
    n_dict_terms = len(tokens)
    dictionary = dict(enumerate(tokens))
    bests = np.random.randint(0, n_dict_terms, n_frames)
    logits = np.zeros((n_dict_terms, n_frames))
    logits[bests, np.arange(n_frames)] = 1.0
    decoded = text.decode_best(torch.tensor(logits).T, dictionary)
    assert decoded == [tokens[i] for i in bests]
