from pathlib import Path

import click

from timething import align as timething_align
from timething import cutter, dataset, job, llm, text, utils  # type: ignore


@click.group()
def cli():
    """Timething is a library for aligning text transcripts with audio.
    Use one of the commands listed below to get started.
    """
    pass


@cli.command()
@click.option(
    "--language",
    default="english",
    show_default=True,
    help="Key in timething/models.yaml.",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True),
    help="Full path to metadata csv.",
)
@click.option(
    "--alignments-dir",
    required=True,
    type=click.Path(exists=True),
    help="Dir to write results to.",
)
@click.option(
    "--batch-size",
    required=True,
    type=int,
    help="Number of examples per batch",
)
@click.option(
    "--n-workers",
    required=True,
    type=int,
    help="Number of worker processes to use",
)
@click.option(
    "--use-gpu",
    type=bool,
    default=True,
    show_default=True,
    help="Use the gpu, if we have one",
)
@click.option(
    "--k-shingles",
    type=int,
    default=5,
    show_default=True,
    help="Number of shingles to use for the partition score",
)
@click.option(
    "--offline",
    type=bool,
    default=False,
    show_default=True,
    help="Offline mode",
)
def align_short(
    language: str,
    metadata: str,
    alignments_dir: str,
    batch_size: int,
    n_workers: int,
    use_gpu: bool,
    k_shingles: int,
    offline: bool,
):
    """Align text transcripts with short audio.

    You provide the audio files, as well as a text file with the complete text
    transcripts. Timething will output a list of time-codes for each word and
    character that indicate when this word or letter was spoken in the audio
    you provided.

    Audio files should be short enough to be processed in a single batch.
    Processing length depends on amount of memory you have available on your
    GPU or main memory. In practice you may want to keep snippets under 10 to
    20 seconds.
    """

    # retrieve the config for the given language
    cfg = utils.load_config(language, k_shingles, local_files_only=offline)

    # construct the dataset
    ds = dataset.SpeechDataset(Path(metadata), cfg.sampling_rate)

    # construct and run the job
    click.echo("setting up aligner...")
    j = job.Job(
        cfg,
        ds,
        batch_size=batch_size,
        n_workers=n_workers,
        gpu=use_gpu,
        output_path=Path(alignments_dir),
    )

    # construct the generic model text cleaner
    ds.clean_text_fn = text.TextCleaner(cfg.language, j.aligner.vocab)

    # go
    click.echo("starting aligment...")
    j.run()


@cli.command()
@click.option(
    "--language",
    default="english",
    show_default=True,
    help="Key in timething/models.yaml.",
)
@click.option(
    "--audio-file",
    required=True,
    type=click.Path(exists=True),
    help="Full path to audio file.",
)
@click.option(
    "--transcript-file",
    required=True,
    type=click.Path(exists=True),
    help="Full path to transcription text file.",
)
@click.option(
    "--alignments-dir",
    required=True,
    type=click.Path(exists=True),
    help="Dir to write results to.",
)
@click.option(
    "--batch-size",
    required=True,
    type=int,
    help="Number of examples per batch",
)
@click.option(
    "--n-workers",
    required=True,
    type=int,
    help="Number of worker processes to use",
)
@click.option(
    "--use-gpu",
    type=bool,
    default=True,
    show_default=True,
    help="Use the gpu, if we have one",
)
@click.option(
    "--k-shingles",
    type=int,
    default=5,
    show_default=True,
    help="Number of shingles to use for the partition score",
)
@click.option(
    "--seconds-per-window",
    type=int,
    default=10,
    show_default=True,
    help="Number of seconds to cut the track into before feeding to the model",
)
@click.option(
    "--offline",
    type=bool,
    default=False,
    show_default=True,
    help="Offline mode",
)
def align_long(
    language: str,
    audio_file: Path,
    transcript_file: Path,
    alignments_dir: str,
    batch_size: int,
    n_workers: int,
    use_gpu: bool,
    k_shingles: int,
    seconds_per_window: int,
    offline: bool,
):
    """Align text transcripts with long audio.

    You provide a single long audio file, as well as a text file with the
    complete text transcript. Timething will output a list of time-codes for
    each word and character that indicate when this word or letter was spoken
    in the audio you provided.

    """

    # retrieve the config for the given language
    cfg = utils.load_config(language, k_shingles, local_files_only=offline)

    # read in the transcript
    with open(transcript_file, "r") as f:
        transcript = f.read()
        transcript = " ".join(transcript.lower().splitlines())

    # construct the dataset
    ds = dataset.WindowedTrackDataset(
        Path(audio_file),
        Path(audio_file).suffix[1:],
        transcript,
        seconds_per_window * 1000,
        seconds_per_window * 1000,
        16000,
    )

    click.echo("setting up aligner...")
    j = job.LongTrackJob(cfg, ds, batch_size=batch_size, n_workers=n_workers)

    click.echo("starting aligment...")
    alignment = j.run()

    click.echo("writing aligment...")
    utils.write_alignment(
        Path(alignments_dir), Path(audio_file).stem, alignment
    )


@cli.command()
@click.option(
    "--from-metadata",
    required=True,
    type=click.Path(exists=True),
    help="Full path to the source dataset metadata csv.",
)
@click.option(
    "--to-metadata",
    required=True,
    help="Full path to the destination dataset metadata csv; will be created.",
)
@click.option(
    "--alignments-dir",
    required=True,
    type=click.Path(exists=True),
    help="Aligments dir for the source dataset.",
)
@click.option(
    "--cut-threshold-seconds",
    default=8.0,
    type=float,
    help="Maximum duration. Source recordings over this will be recut.",
)
@click.option(
    "--pause-threshold-model-frames",
    default=20,
    type=int,
    help="Lowest value of model frames between words before snipping here.",
)
@click.option(
    "--padding-ms",
    default=80,
    type=int,
    help="Relax the cut at the beginning and end by this no. of milliseconds",
)
def recut(
    from_metadata: str,
    to_metadata: str,
    alignments_dir: str,
    cut_threshold_seconds: float,
    pause_threshold_model_frames: int,
    padding_ms: int,
):
    """Recut an existing dataset.

    Sometimes you want smaller files, like when training a machine learning
    model. This command will cut long recordings into smaller ones, such that
    the cut threshold in seconds is never exceeded. Will write out a new
    dataset with split audio and split texts.
    """

    click.echo("starting re-cutting...")
    cutter.dataset_recut(
        Path(from_metadata),
        Path(to_metadata),
        Path(alignments_dir),
        cut_threshold_seconds,
        pause_threshold_model_frames,
        padding_ms,
    )


@cli.command()
@click.option(
    "--language",
    default="english",
    show_default=True,
    help="Key in timething/models.yaml.",
)
def download(language: str):
    """Downloads a speech recognition model and puts it in the cache dir.

    Timething will download the language model you need on demand. In some
    cases, you may want to cache the model before though.
    """

    # retrieve the config for the given language
    cfg = utils.load_config(language)

    # download
    click.echo("dowloading speech recognition model...")
    timething_align.Aligner.build("cpu", cfg)


@cli.command()
@click.option(
    "--transcript-file",
    required=True,
    type=click.Path(exists=True),
    help="Full path to the transcript file.",
)
@click.option(
    "--output-file",
    required=True,
    type=click.Path(exists=False),
    help="Full path to the output file.",
)
@click.option("--openai-api-key", required=True, help="OpenAI API key.")
def clean_transcript(
    transcript_file: str, output_file: str, openai_api_key: str
):
    """Cleans a transcript file.

    Currently passes through chat-gpt.
    """

    click.echo("cleaning transcript file {}...".format(transcript_file))

    chatter = llm.ChatGPT(openai_api_key)
    with open(Path(transcript_file), "r") as f:
        cleaned = text.clean_with_llm(chatter, f.read())

    with open(Path(output_file), "w") as f:
        f.write(cleaned)

    click.echo("wrote cleaned transcript to {}".format(output_file))


if __name__ == "__main__":
    cli(prog_name="timething")  # type: ignore
