import itertools
import re
import typing

import torch
from num2words import num2words  # type: ignore

# detect numbers
NUMS_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

# cleaning


class TextCleaner:
    """A generic text cleaner. Langauge is an ISO 639-1 code
    """

    def __init__(self, language: str, vocab: typing.List[str]):
        self.language = language
        self.allowed_chars = "".join(sorted({c for c in vocab if len(c) == 1}))
        self.blacklist = re.compile(f"[^{re.escape(self.allowed_chars)}]+")

    def __call__(self, text: str):

        # assumes a lower case only text model.
        text = text.casefold()

        text = nums2words(text, lang=self.language)

        # replace all non-vocabulary characters with space
        text = re.sub(self.blacklist, " ", text)

        # collapse repeating spaces and replace with pipe
        text = re.sub(" +", " ", text)
        text = text.strip()
        text = text.replace(" ", "|")

        return text


def nums2words(text: str, lang: str):
    """Replace number occurences with a corresponding text version.

    Includes weak heuristics to identify years numbers, since these are
    pronounced differently.
    """

    def fn(match, to="cardinal"):
        number = float(match.group(0))
        if number.is_integer() and number > 1800 and number < 2100:
            to = "year"
            number = int(number)

        return num2words(number, lang=lang, to=to)

    return re.sub(NUMS_RE, fn, text)


# decoding


def ctc_collapse(tokens: typing.List[str], blank="<pad>", delimiter="|"):
    "Collapse CTC blanks"
    transcript = "".join(c for c, _ in itertools.groupby(tokens))
    return " ".join(transcript.replace(blank, "").split(delimiter))


def decode_best(logprobs, dictionary):
    "Argmax decoding of P(char | audio)."
    x = torch.argmax(logprobs, dim=-1)
    return [dictionary[code.item()] for code in x.squeeze()]


def best_ctc(logprobs, dictionary, blank="<pad>", delimiter="|"):
    "Argmax decoding of P(char | audio). Inclues CTC collapsing"

    tokens = decode_best(logprobs, dictionary)
    return ctc_collapse(tokens, blank, delimiter)


# partitioning


def k_shingle(text: str, k=5):
    "Shingle the text, yielding len k fragments with an overlap of 1."
    return {text[i : i + k] for i in range(len(text)) if i <= len(text) - k}


def jaquard(a: set, b: set) -> float:
    "The Jaquard similarity between two sets"
    len_intersection = len(a.intersection(b))
    len_union = len(a.union(b))
    if len_union != 0.0:
        return len_intersection / len_union
    return 0.0


def similarity(query: str, candidate: str, k: int) -> float:
    "Similarity of two strings. Returns a score >= 0 <= 1."
    return jaquard(k_shingle(query, k), k_shingle(candidate, k))
