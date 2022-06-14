"""
Forced Alignment with Wav2Vec2.
"""

import dataclasses
import typing
from dataclasses import dataclass
from difflib import ndiff

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore

from timething import text


@dataclass
class Config:

    # the id of the hugging face model
    hugging_model: str

    # a hash or tag to pin the model to
    hugging_pin: str

    # sampling rate expected from this model
    sampling_rate: int

    # language code
    language: str

    # k-shingles for partition score
    k_shingles: int


@dataclass
class BestPath:
    """Optimal alignment up to position i_transcript in transcription tokens
    and up to i_frame in audio frames.
    """

    # index into the transcript tokens
    i_transcript: int

    # index into model audio frames
    i_frame: int

    # optimal score up to (i_transcript, i_frame) subproblem
    score: float

    # score for this token at this frame under the CTC model
    frame_score: float


@dataclass
class Segment:
    """A segment of one or more characters, mapped to a single range of audio.
    """

    # the string of characters in the segment
    label: str

    # start offset in model frames in the audio
    start: float

    # end offset in model frames in the audio
    end: float

    # confidence score under the given ASR model
    score: float

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Alignment:

    # example identifier
    id: str

    # log scale probabilities of characters over frames
    scores: np.ndarray

    # asr results with best decoding
    recognised: str

    # optimal alignment
    path: typing.List[BestPath]

    # character segments
    chars_cleaned: typing.List[Segment]

    # character segments without cleaning
    chars: typing.List[Segment]

    # word segments
    words_cleaned: typing.List[Segment]

    # original word segments
    words: typing.List[Segment]

    # number of stft frames in this example
    n_model_frames: int

    # number of audio samples in this example
    n_audio_samples: int

    # the sampling rate
    sampling_rate: int

    # how well do the audio and transcript match in this example
    partition_score: float

    def model_frames_to_fraction(self, n_frames) -> float:
        "Returns the fraction of the padded example at n_frames"
        return n_frames / self.n_model_frames

    def model_frames_to_seconds(self, n_frames) -> float:
        "Returns the absolute offset in seconds at n_frames"
        fraction = self.model_frames_to_fraction(n_frames)
        return fraction * self.n_audio_samples / self.sampling_rate

    def seconds_to_model_frames(self, n_seconds) -> int:
        "Returns the absolute offset in model frames at n_seconds"
        n_total_seconds = self.n_audio_samples / self.sampling_rate
        fraction = n_seconds / n_total_seconds
        return round(fraction * self.n_model_frames)


class Aligner:
    """Align the given transcription to the given audio file.
    """

    def __init__(self, device, processor, model, sr=16000, k_shingles=5):
        self.device = device
        self.processor = processor
        self.model = model
        self.sr = sr
        self.k_shingles = k_shingles

    @staticmethod
    def build(device, cfg: Config):
        return Aligner(
            device,
            Wav2Vec2Processor.from_pretrained(
                cfg.hugging_model, revision=cfg.hugging_pin
            ),
            Wav2Vec2ForCTC.from_pretrained(
                cfg.hugging_model, revision=cfg.hugging_pin
            ).to(device),
            cfg.sampling_rate,
            cfg.k_shingles,
        )

    def align(self, batch) -> typing.List[Alignment]:
        """Align the audio and the transcripts in the batch.

        Returns a list of aligments, one per example. CTC probablities are
        processed in a single batch, on the gpu. Backtracking is performed in
        a loop on the CPU.
        """

        xs, ys, ys_original, ids = batch
        scores_batched = self.logp(xs)
        alignments = []
        for i in range(len(ys)):

            # i_th recording in the batch
            x = xs[i]
            y = ys[i]
            id = ids[i]
            y_original = ys_original[i]
            scores = scores_batched[i].T

            # metadata
            n_model_frames = scores.shape[1]
            n_audio_samples = x.shape[1]

            # asr
            recognised = text.best_ctc(scores, self.dictionary)
            y_whitespace = y.replace("|", " ").strip()
            partition_score = text.similarity(
                recognised.strip(), y_whitespace, self.k_shingles
            )

            # align
            path = best(scores, self.tokens(y), self.blank_id)

            # char results
            chars_cleaned = to_segments(path, y)
            chars = align_clean_text(y, y_original, chars_cleaned)

            # word results
            words_cleaned = to_words(chars_cleaned)
            words = to_words(chars, separator=" ")

            # append to batch
            alignments.append(
                Alignment(
                    id,
                    scores,
                    recognised,
                    path,
                    chars_cleaned,
                    chars,
                    words_cleaned,
                    words,
                    n_model_frames,
                    n_audio_samples,
                    self.sr,
                    partition_score,
                )
            )

        return alignments

    def logp(self, batch):
        """
        Batch processing for log probabilities
        batch: torch.tensor [B, C, T] - batch, channel, time
        """

        batch = self.processor(
            batch, sampling_rate=self.sr, return_tensors="pt", padding=True
        )
        batch = batch.to(self.device)
        input_values = batch.input_values.squeeze(0).squeeze(1)

        with torch.no_grad():
            out = self.model(input_values, attention_mask=batch.attention_mask)
            log_probs = torch.log_softmax(out.logits, dim=-1)

        return log_probs.cpu().detach()

    def tokens(self, y: str):
        v = self.vocab
        return [v[c] for c in y]

    @property
    def vocab(self):
        return self.processor.tokenizer.get_vocab()

    @property
    def dictionary(self):
        return {c: i for i, c in self.vocab.items()}

    @property
    def blank_id(self):
        return self.vocab[text.BLANK_TOKEN]

    @property
    def sos_id(self):
        return self.vocab[text.SOS_TOKEN]


def best(
    scores: np.ndarray, tokens: typing.List[int], blank_id: int
) -> typing.List[BestPath]:
    """Return the optimal alignment score of `tokens` aligned vs `scores`.

    This is a bottom up dynamic programing implementation of the following
    recurrence:

    ```
    best(i_text, 0) = scores[text[i_text], 0]
    best(i_text, i_frame) = max(
        best(i_text, i_frame - 1) * scores[blank_id, i_frame]
        best(i_text, i_frame - 1) * scores[text[i_text], i_frame]
    )
    ```

    With the provision:

    ```
    best(i_text > i_frame or i_text < 0) = -np.inf
    ```

    Arguments:

      scores: a `n_tokens` x `n_frames` table where each element represents
              the score of a given token at a given frame. Scores are log
              probabilities.
      tokens: a list of character ids representing a transcription to align to
              `scores`. Numbers are offsets into the rows of `scores`.

    Returns:

      best: An optimal alignment.
    """

    n_tokens, n_frames = scores.shape
    n_token_chars = len(tokens)
    bests = np.ones((n_token_chars, n_frames)) * -np.inf
    for to_frame in range(n_frames):
        for to_char in range(n_token_chars):
            if to_char > to_frame or to_char < 0:
                continue

            # fmt: off
            i_char = tokens[to_char]
            if to_frame == 0:
                best = scores[i_char, to_frame]
            else:
                best = max([
                    bests[to_char, to_frame - 1] + scores[blank_id, to_frame],
                    bests[to_char - 1, to_frame - 1] + scores[i_char, to_frame]
                ])

            # fmt: on
            bests[to_char, to_frame] = best

    # backtrack to reconstruct the path. this way we don't have to save
    # anything in the forward pass.
    path: typing.List[BestPath] = []
    t = int(np.argmax(bests[:, -1]))
    for f in range(n_frames - 1, -1, -1):
        score = np.exp(bests[t, f])
        frame_score = np.exp(scores[tokens[t], f])
        path.insert(0, BestPath(t, f, score, frame_score))
        if bests[t - 1, f - 1] > bests[t, f - 1]:
            t -= 1

    return path


def to_segments(
    path: typing.List[BestPath], transcript: str
) -> typing.List[Segment]:
    """Convert the path into a list of collapsed segments. Merges path
    elements with the same transcript index into a single `Segment`; e.g.
    pieces where characters were padded with BLANK.
    """

    i_transcript = None
    start, end = 0, 0
    score = 0.0
    segments: typing.List[Segment] = []

    def emit():
        if i_transcript is not None:
            segments.append(
                Segment(transcript[i_transcript], start, end, score)
            )

    for el in path:
        if el.i_transcript == i_transcript:
            # accumulate
            score += el.score
            end = el.i_frame
        else:
            # emit and start
            emit()
            i_transcript = el.i_transcript
            start = el.i_frame
            end = el.i_frame
            score = el.score

    # last
    emit()

    return segments


def to_words(
    segments: typing.List[Segment], separator="|"
) -> typing.List[Segment]:
    """Merge Segments on word boundaries. A single input segment can include
    multiple characters, and may end with a separator.
    """

    word = ""
    start, end = 0.0, 0.0
    score = 0.0
    words: typing.List[Segment] = []

    def emit():
        if word:
            words.append(Segment(word, start, end, score))

    for s in segments:
        if s.label.endswith(separator):
            # emit
            if s.label != separator:
                # start
                word += s.label.rstrip(separator)
                end = s.end
                score += s.score

            emit()
            word = ""
        else:
            # accumulate
            if not word:
                start = s.start
                score = 0.0

            word += s.label
            end = s.end
            score += s.score

    # last
    emit()

    return words


def align_clean_text(
    in_text: str, out_text: str, in_segs: typing.List[Segment], separator="|"
) -> typing.List[Segment]:
    """Timething TTS models align on cleaned texts. In order to show
    alignments in terms of the (uncleaned) input text, we have to match the
    cleaned string to the input string. This match can then be used to impute
    input text timecodes from the cleaned text timecodes.

    Arguments:

    in_text: str
        The cleaned input string
    out_text: str
        The original, uncleaned input string
    in_segs: typing.List[Segment]
        Segmentation of the cleaned input string. Segmented on character level

    Returns:

    out_segs: typing.List[Segment]
        Segmentation of the original un-cleaned input string, with adjusted
        symbols and timecodes.
    """

    def clone(x, **changes):
        return dataclasses.replace(x, **changes)

    if not in_text:
        return []

    # make ndiff's life a bit easier. We can recover the original characters
    # from the out_text string, since we always maintain an index into it.
    out_text_normalised = out_text.lower().replace(" ", separator)

    # set up main loop
    out_segs: typing.List[Segment] = []
    i, j = 0, 0  # in_text[i], out_text[j]
    in_seg, out_seg = None, None
    edit_seg = None  # accrue edit segs here
    leading_additions = ""  # leading with one or more additions
    for d in diff(in_text, out_text_normalised):
        # the TextCleaner uses text.casefold(). This lower-cases, but also
        # normalises unicode, s.t. e.g. ß becomes ss. Since we don't want to
        # change the number of characters here, we're just downcasing.
        op = d[0]

        if op != "+":
            # moved in in_text
            i += 1

        if op != "-":
            # moved in out_text
            j += 1

        in_seg = in_segs[i - 1]
        out_char = out_text[j - 1]

        if op == " " or op == "?":
            if edit_seg:
                out_segs.append(edit_seg)
                edit_seg = None
            if leading_additions:
                out_char = leading_additions + out_char
                leading_additions = ""
            out_seg = clone(in_seg, label=out_char)
            out_segs.append(out_seg)
        else:
            if op == "-":
                if edit_seg:
                    edit_seg.end = in_seg.end
                else:
                    edit_seg = clone(in_seg, label="")
            if op == "+":
                if edit_seg:
                    edit_seg.label += out_char
                elif out_seg:
                    out_seg.label += out_char
                else:
                    # we started with a +
                    leading_additions += out_char

    if edit_seg and edit_seg.label:
        out_segs.append(edit_seg)

    # invariants
    assert i == len(in_text)
    assert j == len(out_text)

    return out_segs


def diff(a, b: str):
    """Like difflib.ndiff, but streches of [+-]* are stably sorted to always
    have deletions first, then inserts.
    """

    deletions = []
    inserts = []
    for d in ndiff(a, b):
        op = d[0]
        if op == "+":
            inserts.append(d)
            continue
        elif op == "-":
            deletions.append(d)
            continue
        elif deletions or inserts:
            while deletions:
                yield deletions.pop(0)
            while inserts:
                yield inserts.pop(0)
        yield d

    while deletions:
        yield deletions.pop(0)
    while inserts:
        yield inserts.pop(0)
