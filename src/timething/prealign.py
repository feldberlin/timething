import itertools
import typing

import torch


def decode_best(logprobs, vocab, delimiter="|"):
    "Argmax decoding of P(char | audio). Accepts a B, T Tensor"

    transcripts = []
    d = {v: k for (k, v) in vocab.items()}
    x = torch.argmax(logprobs, dim=2)
    for i in range(x.shape[0]):  # loop over batch dimension
        tokens = [d[code.item()] for code in x[i].squeeze()]
        transcript = "".join(c for c, _ in itertools.groupby(tokens))
        transcript = " ".join(transcript.replace(d[0], "").split("|"))
        transcripts.append(transcript)

    return transcripts


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

    prediction: the query string we would like to find
    transcription: the dataset we are searching in, e.g. the full transcript
    n_chars: the number of characters in each window

    Return the (i, j) (query, candidate) pairs with a larger than threshold
    similarity.

    The query is split into a number of windows. The transcription is also
    split into windows. We then calculate the Jaquard similarity between each
    query window and each transcription window.

    i and j are indices into the
    """

    queries = windows(prediction.lower(), n_chars=n_chars)
    candidates = windows(transcription.lower(), n_chars=n_chars)
    for i, query in enumerate(queries):
        for j, candidate in enumerate(candidates):
            similarity = jaquard(k_shingle(query), k_shingle(candidate))
            if similarity >= threshold:
                yield i, j, similarity
