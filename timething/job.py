from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from timething import align, utils  # type: ignore


class Job:
    """
    An alignment job
    """

    def __init__(
        self,
        cfg: align.Config,
        loader: DataLoader,
        device: torch.device,
        output_path: Path,
    ):
        self.cfg = cfg
        self.loader = loader
        self.device = device
        self.output_path = output_path
        self.aligner = align.Aligner.build(device, cfg)

    def run(self):
        total = len(self.loader)
        for i, (xs, ys, ids) in tqdm(enumerate(self.loader), total=total):
            alignments = self.aligner.align((xs, ys))

            # write the alignments
            for i, id in enumerate(ids):
                utils.write_alignment(self.output_path, id, alignments[i])
