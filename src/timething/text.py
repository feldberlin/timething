import re
import typing

from num2words import num2words  # type: ignore

# detect numbers
NUMS_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def clean_text_fn(language: str, vocab: typing.List[str]):
    """
    A generic text cleaner. Langauge is an ISO 639-1 code
    """

    allowed_chars = "".join(sorted({c for c in vocab if len(c) == 1}))
    allowed_chars_re = re.compile(f"[^{re.escape(allowed_chars)}]+")

    def fn(text: str):

        # assumes a lower case only text model.
        text = text.casefold()

        text = nums2words(text, lang=language)

        # replace all non-vocabulary characters with space
        text = re.sub(allowed_chars_re, " ", text)

        # collapse repeating spaces and replace with pipe
        text = re.sub(" +", " ", text)
        text = text.strip()
        text = text.replace(" ", "|")

        return text

    return fn


def nums2words(text: str, lang: str):
    """
    Replace number occurences with a corresponding text version.
    """

    def fn(match):
        return num2words(float(match.group(0)), lang=lang)

    return re.sub(NUMS_RE, fn, text)
