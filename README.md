[![Build](https://github.com/feldberlin/timething/workflows/CI/badge.svg)](https://github.com/feldberlin/timething/actions)

# Timething

Timething is a library for aligning text transcripts with audio. You provide
an audio file, as well as a text file with the complete text transcript.
Timething will output a list of time-codes for each word and character that
indicate when this word or letter was spoken in the audio you provided.
Timething strives to be fast and accurate, and can run on both GPUs or CPUs.

Timething uses uses powerful Wav2Vec based speech recognition models hosted by
the Hugging Face AI community. The approach is described in this [PyTorch
Tutorial](https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html),
as well as in this [paper](https://arxiv.org/abs/2007.09127).

## Running

Timething currently expects to find a folder containing one or more chapters
in the following form:


    └── audio/
        ├── metadata.csv
        ├── alignments/
        └── wavs/
            ├── chapter01.wav
            ├── chapter02.wav
            └── chapter03.wav


The file `metadata.csv` should contain one entry per wav file in the following
format:

```csv
chapter01|The transcript for chapter01 on a single line here
chapter02|The transcript for chapter02 on a single line here
chapter03|The transcript for chapter03 on a single line here
```

You can now run timething on your CPU or GPU:


```bash
python cli.py \
  --model german \
  --metadata audio/metadata.csv \
  --alignments-dir audio/alignments \
  --batch-size 8 \
  --n-workers 8
```

Results will be written into the `alignments` folder, into a single file json
file named after each audio id. Each file will contain the character level and
the word level alignments. For word level alignments, each word will have the
starting time in seconds, the ending time in seconds, the confidence level for
that word, and the word label. For character level alignments we have the same
thing, except for characters.

## Supported languages

Currently supported languages can be found [in
models.yaml](https://github.com/feldberlin/timething/blob/main/timething/models.yaml)

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
