from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from timething import job, align, utils  # type: ignore


@click.command()
@click.option("--model",  required=True,
              help="Key in timething/models.yaml.")
@click.option("--metadata", required=True, type=click.Path(),
              help="Full path to metadata csv.")
@click.option("--alignments-dir", required=True, type=click.Path(),
              help="Dir to write results to.")
@click.option("--n-workers", required=True, type=int,
              help="Number of worker processes to use", )
def main(model: str, metadata: str, alignments_dir: str, n_workers: int):

    # retrieve the config for the given model
    cfg = utils.load_config(model)

    # construct the dataset
    ds = align.SpeechDataset(
        Path(metadata), cfg.sampling_rate, clean_text_fn=align.clean_text_fn
    )

    # load from the dataset
    loader = DataLoader(
        ds, n_workers, collate_fn=align.collate_fn, shuffle=False
    )

    # use a gpu if it's there
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # construct and run the job
    j = job.Job(cfg, loader, device, Path(alignments_dir))
    j.run()


if __name__ == "__main__":
    main()
