from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from timething import dataset, job, utils  # type: ignore


@click.command()
@click.option("--model", required=True, help="Key in timething/models.yaml.")
@click.option(
    "--metadata",
    required=True,
    type=click.Path(),
    help="Full path to metadata csv.",
)
@click.option(
    "--alignments-dir",
    required=True,
    type=click.Path(),
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
def main(
    model: str,
    metadata: str,
    alignments_dir: str,
    batch_size: int,
    n_workers: int,
):

    # retrieve the config for the given model
    cfg = utils.load_config(model)

    # construct the dataset
    ds = dataset.SpeechDataset(Path(metadata), cfg.sampling_rate)

    # load from the dataset
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=n_workers,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    # use a gpu if it's there
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # construct and run the job
    print("setting up aligment job...")
    j = job.Job(cfg, loader, device, Path(alignments_dir))

    # construct the generic model text cleaner
    ds.clean_text_fn = dataset.clean_text_fn(j.aligner.vocab())

    # go
    print("starting aligment job...")
    j.run()


if __name__ == "__main__":
    main()
