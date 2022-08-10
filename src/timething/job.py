import typing
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # type: ignore

from timething import align, dataset, utils  # type: ignore


class Job:
    """An alignment job.
    """

    def __init__(
        self,
        cfg: align.Config,
        ds: Dataset,
        batch_size: int,
        n_workers: int,
        gpu: bool,
        output_path: typing.Optional[Path],
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
        ret = []
        for xs, ys, ys_original, ids in tqdm(self.loader, total=total):
            alignments = self.aligner.align((xs, ys, ys_original, ids))
            ret.append(alignments)

            # write the alignments
            if self.output_path:
                for i, id in enumerate(ids):
                    utils.write_alignment(self.output_path, id, alignments[i])

        return ret
