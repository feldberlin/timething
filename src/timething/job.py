from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # type: ignore

from timething import align, dataset, utils  # type: ignore


class Job:
    """
    An alignment job
    """

    def __init__(
        self,
        cfg: align.Config,
        ds: Dataset,
        output_path: Path,
        batch_size: int,
        n_workers: int,
        gpu: bool,
    ):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
        self.output_path = output_path
        self.aligner = align.Aligner.build(self.device, cfg)
        self.loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=n_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )

    def run(self):
        total = len(self.loader)
        for xs, ys, ys_original, ids in tqdm(self.loader, total=total):
            alignments = self.aligner.align((xs, ys, ys_original))

            # write the alignments
            for i, id in enumerate(ids):
                utils.write_alignment(self.output_path, id, alignments[i])
