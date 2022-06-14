"""
Forced Alignment with Wav2Vec2.

# Scores

In the case of the dynamic programming alignment algorithm implemented in
`best`, a score is the (log scale) probability the best path up to a point, or
the (log scale) probability of a character appearing in a given frame.

Scores may be used to compare the quality of alignments. In order for this to
work well, there should be a top level score for an aligned (text, audio)
pair. These scores should be comparable across different pairs. Different
lengths of transcripts and audio files should lead to normalised scores which
can be compared. This can be done by taking a mean probability value. We will
use the geometric mean here.

"""

import dataclasses
import math
import typing
from dataclasses import dataclass
from difflib import ndiff
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore

from timething import text

# default cache dir path
CACHE_DIR_DEFAULT = Path("~/.cache/timething").expanduser()


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

    # cache dir for models
    cache_dir: Path = CACHE_DIR_DEFAULT

    # currently needed for hf to work offline
    local_files_only: bool = False


@dataclass
class BestPath:
    """Optimal alignment up to position i_transcript in transcription tokens
    and up to i_frame in audio frames.
    """

    # index into the transcript tokens
    i_transcript: int

    # index into model audio frames
    i_frame: int

    # optimal (non-log) probability up to (i_transcript, i_frame) subproblem.
    path_score: float

    # (non-log) probability for this token at this frame under the CTC model.
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

    # product of the asr frame probabilities along this segment.
    score: float

    # geometric mean of the asr frame probabilities along this segment
    geometric_score: float

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

    # normalised score of this alignment
    alignment_score: float

    # probability of this alignment
    alignment_probability: float

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
                cfg.hugging_model,
                revision=cfg.hugging_pin,
                cache_dir=str(cfg.cache_dir),
                local_files_only=cfg.local_files_only,
            ),
            Wav2Vec2ForCTC.from_pretrained(
                cfg.hugging_model,
                revision=cfg.hugging_pin,
                cache_dir=str(cfg.cache_dir),
                local_files_only=cfg.local_files_only,
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

            # top level scores
            alignment_probability = path[-1].path_score
            alignment_score = geometric_mean(
                np.prod([c.geometric_score for c in chars_cleaned]),
                len(chars_cleaned),
            )

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
                    alignment_score,
                    alignment_probability,
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
        "Start and end of sentence tokens included"
        v = self.vocab
        return [self.sos_id] + [v[c] for c in y]

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
        best(i_text - 1, i_frame - 1) * scores[text[i_text], i_frame]
    )
    ```

    With the provision:

    ```
    best(i_text > i_frame or i_text < 0) = -np.inf
    ```

    Caveats: this implementation is currently not vectorised with torch.

    Arguments:

      scores: a `n_vocab` x `n_frames` table where each element represents the
              score of a given token at a given frame. Scores are log
              probabilities. `n_vocab` is the number of characters in the
              alphabet we are using.
      tokens: a list of character ids representing a transcription to align to
              `scores`. Numbers are offsets into the rows of `scores`. Tokens
              has a length of n_token_chars.

    Returns:

      best: An optimal alignment, which is `n_frames` in length.
    """

    n_vocab, n_frames = scores.shape
    n_token_chars = len(tokens)

    # builds a trellis of n_token_chars x n_frames.
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

    # backtrack to reconstruct the path. This is an optimisation so that we
    # don't have to save anything in the forward pass. we can start with the
    # the best transcription offset to start with on the last frame. then
    # we look at the two possible previous elements in bests. the higher one
    # is the one we came from. this also determines if the current one is
    # a repeat or not.
    path: typing.List[BestPath] = []

    # start at the end of the transcript; this is *force* aligned
    t = n_token_chars - 1

    # best frame for last transcript character.
    best_starting_frame_idx = int(np.argmax(bests[-1, :]))

    # loop backwards through the frames.
    # note: range(3, -1, -1) => [3, 2, 1, 0]
    for f in range(best_starting_frame_idx, -1, -1):

        # did we move in the transcript?
        shifted = bests[t - 1, f - 1] > bests[t, f - 1]

        # probability of the path up to here
        path_score = math.exp(bests[t, f])

        # probability of this transcript char at this audio frame
        frame_score = math.exp(scores[tokens[t] if shifted else blank_id, f])

        # record it
        path.insert(0, BestPath(t, f, path_score, frame_score))

        # shifted? if so move transcript pointer `t`. the last iteration
        # points to a negative frame number, so no one off issue
        if shifted:
            t -= 1

    # remove start of sentence token from path
    path = [
        dataclasses.replace(x, i_transcript=x.i_transcript - 1)
        for x in path
        if x.i_transcript > 0
    ]

    return path


def to_segments(
    path: typing.List[BestPath], transcript: str, eps=1e-200
) -> typing.List[Segment]:
    """Convert the path into a list of collapsed segments. Merges path
    elements with the same transcript index into a single `Segment`; e.g.
    pieces where characters were padded with BLANK.
    """

    i_transcript = None  # index into the transcript
    start, end = 0, 0  # start and end of this segment in frames
    score = 0.0  # product of the scores in this segment
    i_scores = 0  # number of scores multiplied in the current segment
    segments: typing.List[Segment] = []

    def emit():
        if i_transcript is not None:
            segments.append(
                Segment(
                    transcript[i_transcript],
                    start,
                    end,
                    math.exp(score),
                    geometric_mean(math.exp(score), i_scores),
                )
            )

    for el in path:
        if el.i_transcript == i_transcript:
            # accumulate
            end = el.i_frame
            score += np.log(el.frame_score + eps)
            i_scores += 1
        else:
            # emit and start
            emit()
            i_transcript = el.i_transcript
            start = el.i_frame
            end = el.i_frame
            score = np.log(el.frame_score + eps)
            i_scores = 0

    # last
    emit()

    return segments


def to_words(
    segments: typing.List[Segment], separator="|", eps=1e-200
) -> typing.List[Segment]:
    """Merge Segments on word boundaries. A single input segment can include
    multiple characters, and may end with a separator.
    """

    word = ""  # current word we are constructing
    start, end = 0.0, 0.0  # start and end of this word in frames
    score = 0.0  # product of the framewise probabilities
    i_scores = 0  # number of scores multipled in the current segment
    words: typing.List[Segment] = []  # list of words to return

    def emit():
        if word:
            words.append(
                Segment(
                    word,
                    start,
                    end,
                    math.exp(score),
                    geometric_mean(math.exp(score), i_scores),
                )
            )

    for s in segments:
        if s.label.endswith(separator):
            # emit
            if s.label != separator:
                # start
                word += s.label.rstrip(separator)
                end = s.end
                score += np.log(s.score + eps)
                i_scores += 1

            emit()
            word = ""
        else:
            # accumulate
            if not word:
                start = s.start
                score = 0.0
                i_scores = 0

            word += s.label
            end = s.end
            score += np.log(s.score + eps)
            i_scores += 1

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


# maths


def geometric_mean(product, n_items):
    if not n_items:
        return product
    return math.pow(product, 1.0 / n_items)
