import re
import typing

from num2words import num2words  # type: ignore

# detect numbers
NUMS_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


class TextCleaner:
    """
    A generic text cleaner. Langauge is an ISO 639-1 code
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
    """
    Replace number occurences with a corresponding text version.
    """

    def fn(match):
        return num2words(float(match.group(0)), lang=lang)

    return re.sub(NUMS_RE, fn, text)
