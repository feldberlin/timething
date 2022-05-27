import itertools
import typing

import torch


def decode_best(logprobs, vocab, delimiter="|"):
    "Argmax decoding of P(char | audio)."

    d = {v: k for (k, v) in vocab.items()}
    x = torch.argmax(logprobs, dim=2)
    tokens = [d[code.item()] for code in x.squeeze()]
    transcript = "".join(c for c, _ in itertools.groupby(tokens))
    return " ".join(transcript.replace(d[0], "").split("|"))


def windows(text: str, n_chars: int) -> typing.List[str]:
    """Curt a single text into overlapping windows.

    Each element is a string of length `n_chars`, and each window overlaps by
    half.
    """

    n = int(2 * len(text) / n_chars)

    def offset(i):
        return int(i * n_chars / 2)

    return [text[offset(i) : (offset(i) + n_chars)] for i in range(n)]


def k_shingle(text: str, k=5):
    "Shingle the text, yielding len k fragments with an overlap of 1."
    return {text[i : i + k] for i in range(len(text))}


def jaquard(a: set, b: set) -> float:
    "The Jaquard similarity between two sets"
    return len(a.intersection(b)) / len(a.union(b))


def similarity(prediction: str, transcription: str, n_chars=80, threshold=0.4):
    """Calculate the full product of queries vs canditates.

    Return the (i, j) (query, candidate) pairs with a larger than threshold
    similarity.
    """

    queries = windows(prediction.lower(), n_chars=n_chars)
    candidates = windows(transcription.lower(), n_chars=n_chars)
    for i, query in enumerate(queries):
        for j, candidate in enumerate(candidates):
            similarity = jaquard(k_shingle(query), k_shingle(candidate))
            if similarity > threshold:
                yield i, j, similarity
