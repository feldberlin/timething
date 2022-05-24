"""
Forced Alignment with Wav2Vec2

Adapted from the pytorch website. Original Author: Moto Hira <moto@fb.com>
"""

import dataclasses
import typing
from dataclasses import dataclass
from difflib import ndiff

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore


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


@dataclass
class Segment:
    """
    A segment of one or more characters, mapped to a single range of audio.
    Includes a confidence score under the ASR model.
    """

    # the string of characters in the segment
    label: str

    # start offset in model frames in the audio
    start: int

    # end offset in model frames in the audio
    end: int

    # confidence score under the given ASR model
    score: float

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Point:
    """
    A single point on the alignment path. Used in backtracking
    """

    # point index on the alphabet axis
    token_index: int

    # point index on the time axis
    time_index: int

    # score under the given ASR model
    score: float


@dataclass
class Alignment:

    # log scale probabilities of characters over frames
    log_probs: np.ndarray

    # hmm-like trellis for viterbi
    trellis: np.ndarray

    # path found through the trellis
    path: np.ndarray

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

    def model_frames_to_fraction(self, n_frames):
        "Returns the fraction of the padded example at n_frames"
        return n_frames / self.n_model_frames

    def model_frames_to_n_samples(self, n_frames):
        "Returns the absolute offset in seconds at n_frames"
        fraction = self.model_frames_to_fraction(n_frames)
        return int(fraction * self.n_audio_samples)

    def model_frames_to_seconds(self, n_frames):
        "Returns the absolute offset in seconds at n_frames"
        fraction = self.model_frames_to_fraction(n_frames)
        return fraction * self.n_audio_samples / self.sampling_rate

    def seconds_to_model_frames(self, n_seconds):
        "Returns the absolute offset in model frames at n_seconds"
        n_total_seconds = self.n_audio_samples / self.sampling_rate
        fraction = n_seconds / n_total_seconds
        return round(fraction * self.n_model_frames)


class Aligner:
    """
    Align the given transcription to the given audio file.
    """

    def __init__(self, device, processor, model, sr=16000):
        self.sr = sr
        self.device = device
        self.processor = processor
        self.model = model

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
        )

    def align(self, batch) -> typing.List[Alignment]:
        """
        Align the audio and the transcripts in the batch. Returns a list of
        aligments, one per example. CTC probablities are processed in a single
        batch, on the gpu. Backtracking is performed in a loop on the CPU.
        """

        xs, ys, ys_original = batch
        log_probs = self.logp(xs)
        alignments = []
        for i in range(len(ys)):
            x = xs[i]
            y = ys[i]
            y_original = ys_original[i]
            log_prob = log_probs[i]
            tokens = self.tokens(y)
            trellis = build_trellis(log_prob, tokens)
            path = backtrack(trellis, log_prob, tokens)
            chars_cleaned = merge_repeats(path, y)
            chars = align_clean_text(y, y_original, chars_cleaned)
            words_cleaned = merge_words(chars_cleaned)
            words = merge_words(chars, separator=" ")
            n_model_frames = trellis.shape[0] - 1
            n_audio_samples = x.shape[1]
            alignment = Alignment(
                log_probs,
                trellis,
                path,
                chars_cleaned,
                chars,
                words_cleaned,
                words,
                n_model_frames,
                n_audio_samples,
                self.sr,
            )

            alignments.append(alignment)

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

    def vocab(self):
        return self.processor.tokenizer.get_vocab()

    def dictionary(self):
        return {c: i for i, c in enumerate(self.vocab())}

    def tokens(self, y: str):
        d = self.dictionary()
        return [d[c] for c in y]


def build_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):

        # 1. Figure out if the current position was stay or change
        # `emission[J-1]` is the emission at time frame `J` of trellis dim.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = (
            emission[t - 1, tokens[j - 1] if changed > stayed else 0]
            .exp()
            .item()
        )
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments, separator="|") -> typing.List[Segment]:
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label.endswith(separator):
            if i1 != i2:
                segs = segments[i1 : i2 + 1]
                word = "".join([seg.label for seg in segs]).rstrip(separator)
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )
                words.append(
                    Segment(
                        word, segments[i1].start, segments[i2 - 1].end, score
                    )
                )
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def align_clean_text(
    in_text: str, out_text: str, in_segs: typing.List[Segment],
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

    out_segs: typing.List[Segment] = []
    i, j = 0, 0  # in_text[i], out_text[j]
    in_seg, out_seg = None, None
    edit_seg = None  # accrue edit segs here
    leading_additions = ""  # leading with one or more additions
    for d in diff(in_text, out_text.lower()):
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
