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

## Alternatives

There are many mature libraries that can already do forced alignment like
Timething, e.g. the Montreal forced aligner or Aeneas. One list of tools is
maintained [here](https://github.com/pettarin/forced-alignment-tools)

## Thanks

Thanks to [why do birds](http://www.whydobirds.de) for allowing the initial
work on this library to be open sourced.
