[![Build](https://github.com/feldberlin/timething/workflows/CI/badge.svg)](https://github.com/feldberlin/timething/actions)
[![PyPI version](https://badge.fury.io/py/timething.svg)](https://badge.fury.io/py/timething)

# Timething

Timething is a library for aligning text transcripts with audio. You provide
an audio file, as well as a text file with the complete text transcript.
Timething will output a list of time-codes for each word and character that
indicate when this word or letter was spoken in the audio you provided.
Timething strives to be fast and accurate, and can run on both GPUs or CPUs.

Timething uses powerful Wav2Vec based speech recognition models hosted by the
Hugging Face AI community. The approach is described in this [PyTorch
Tutorial](https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html),
as well as in this [paper](https://arxiv.org/abs/2007.09127).

## Installation

To install Timething, you'll need an installation of Python 3.7 or 3.8. You
can then install it using pip:

```bash
pip install timething
```

## Running

Timething currently expects to find a folder containing one or more chapters
in the following form:


    └── dir/
        ├── text.csv
        ├── aligned/
        └── audio/
            ├── chapter01.mp3
            ├── chapter02.mp3
            └── chapter03.mp3


Timething can process many audio formats, including MP3, WAV, FLACC and
OGG/VORBIS.

The file `text.csv` should contain one entry per wav file in the following
format:

```csv
audio/chapter01.mp3|The transcript for chapter01 on a single line here
audio/chapter02.mp3|The transcript for chapter02 on a single line here
audio/chapter03.mp3|The transcript for chapter03 on a single line here
```

You can now run Timething on your CPU or GPU, for example:

```bash
timething --metadata text.csv --alignments-dir aligned
```

You can also specify more options, e.g.:

```bash
timething \
  --model german \
  --metadata text.csv \
  --alignments-dir aligned \
  --batch-size 8 \
  --n-workers 8
```

Run `timething --help` for a full description.

Results will be written into the given folder, e.g. `aligned`. They will be
written into a single json file named after each audio id. Each file will
contain the character level and the word level alignments. For word level
alignments, each word will have the starting time in seconds, the ending time
in seconds, the confidence level for that word and the word label. Character
level alignments have the corresponding results.

You can find an example dataset with alignments output in
[`fixtures/`](https://github.com/feldberlin/timething/blob/main/fixtures).
Here's what the alignment for "one.mp3", which contains only the word "one",
looks like:

```json
{
    "n_model_frames": 72,
    "n_audio_samples": 23392,
    "sampling_rate": 16000,
    "chars": [
        {
            "label": "O",
            "start": 0.5888611111111111,
            "end": 0.6497777777777777,
            "score": 0.9999777873357137
        },
        {
            "label": "n",
            "start": 0.6497777777777777,
            "end": 0.7106944444444444,
            "score": 0.99994424978892
        },
        {
            "label": "e!",
            "start": 0.7106944444444444,
            "end": 0.731,
            "score": 0.9999799728393555
        }
    ],
    "chars_cleaned": [
        {
            "label": "o",
            "start": 0.5888611111111111,
            "end": 0.6497777777777777,
            "score": 0.9999777873357137
        },
        {
            "label": "n",
            "start": 0.6497777777777777,
            "end": 0.7106944444444444,
            "score": 0.99994424978892
        },
        {
            "label": "e",
            "start": 0.7106944444444444,
            "end": 0.731,
            "score": 0.9999799728393555
        }
    ],
    "words": [
        {
            "label": "One!",
            "start": 0.5888611111111111,
            "end": 0.731,
            "score": 0.9999637263161796
        }
    ],
    "words_cleaned": [
        {
            "label": "one",
            "start": 0.5888611111111111,
            "end": 0.731,
            "score": 0.9999637263161796
        }
    ]
}
```

## Supported languages

Currently supported languages can be found [in
models.yaml](https://github.com/feldberlin/timething/blob/main/src/timething/models.yaml).
This currently includes English, German, Dutch, Polish, Italian, Portuguese,
Spanish, French, Russian, Japanese, Greek and Arabic models. We have only
tested the German model so far.

Due to the large number of CTC speech models available on the Hugging Face AI
community, new languages can be easily added to Timething. Alternatively,
Wav2Vec can be fine-tuned as described
[here](https://huggingface.co/blog/fine-tune-wav2vec2-english), using any of
the [Common Voice](https://commonvoice.mozilla.org/en/languages) languages, 87
at the time of writing.

Support for text cleaning is currently minimal, and may need to be extended
for new languages.

## Alternatives

There are many mature libraries that can already do forced alignment like
Timething, e.g. the Montreal forced aligner or Aeneas. One list of tools is
maintained [here](https://github.com/pettarin/forced-alignment-tools).

## Thanks

Thanks to [why do birds](http://www.whydobirds.de) for allowing the initial
work on this library to be open sourced.
