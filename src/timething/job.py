import typing
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # type: ignore

from timething import align, dataset, text, utils  # type: ignore


class Job:
    """An alignment job."""

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
        self.device = utils.best_device()
        self.output_path = output_path
        self.aligner = align.Aligner.build(self.device, cfg)
        self.loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=n_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=self.device != "cpu",
            shuffle=False,
        )

    def run(self):
        total = len(self.loader)
        ret = []
        for xs, ys, ys_original, ids in tqdm(self.loader, total=total):
            alignments = self.aligner.align_batch((xs, ys, ys_original, ids))
            ret.append(alignments)

            # write the alignments
            if self.output_path:
                for i, id in enumerate(ids):
                    utils.write_alignment(self.output_path, id, alignments[i])

        return ret


class LongTrackJob:
    """
    An alignment job for a long track, e.g. 30 minutes of audio.
    """

    def __init__(
        self,
        cfg: align.Config,
        ds: dataset.WindowedTrackDataset,
        batch_size: int,
        n_workers: int,
    ):
        self.cfg = cfg
        self.ds = ds
        self.device = utils.best_device()
        self.aligner = align.Aligner.build(self.device, cfg)
        self.loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=n_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=self.device != "cpu",
            shuffle=False,
        )

        # set the cleaner with the vocab from the aligner
        cleaner = text.TextCleaner(cfg.language, self.aligner.vocab)
        self.ds.set_cleaner(cleaner)

    def run(self):
        total = len(self.loader)
        logit_chunks = []
        for xs, ys, ys_original, ids in tqdm(self.loader, total=total):
            logs = self.aligner.logp(xs)
            logit_chunks.append(logs)

        logits = torch.concat(logit_chunks)
        self.logits = logits.reshape(-1, logits.size(2)).unsqueeze(0)

        print("aligning...")
        # return the alignments
        return self.aligner.align(
            self.logits, [self.ds.cleaned_transcript], [self.ds.transcript], [1]
        )[0]
