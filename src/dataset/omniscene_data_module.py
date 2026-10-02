"""Fixed-batch OmniScene loading, with deterministic resume independent of prefetch."""

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Sampler

from .data_module import worker_init_fn
from .dataset_omniscene import DatasetOmniScene


class ResumableShuffleSampler(Sampler):
    """An infinite sequence of shuffled epochs, indexed by completed updates.

    No iterator/prefetch cursor is checkpointed: only *consumed* samples count.
    With batch=1 and accumulation=1 this is exactly Lightning's global_step.
    """

    def __init__(self, size: int, seed: int, start_step: int = 0):
        self.size, self.seed, self.start_step = size, seed, start_step

    def __iter__(self):
        epoch, offset = divmod(self.start_step, self.size)
        while True:
            order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            yield from order[offset:]
            epoch, offset = epoch + 1, 0


class OmniSceneDataModule(LightningDataModule):
    def __init__(self, dataset_cfg, loader_cfg, test_cfg, start_step=0):
        super().__init__()
        self.dataset_cfg, self.loader_cfg, self.test_cfg = dataset_cfg, loader_cfg, test_cfg
        self.start_step = start_step

    def _loader(self, stage, split="total"):
        cfg = getattr(self.loader_cfg, stage)
        dataset = DatasetOmniScene(self.dataset_cfg, stage, split=split,
                                  compute_pcc=self.test_cfg.compute_pcc)
        sampler = ResumableShuffleSampler(len(dataset), cfg.seed or 0, self.start_step) if stage == "train" else None
        return DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                          num_workers=cfg.num_workers,
                          persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
                          pin_memory=True, worker_init_fn=worker_init_fn,
                          generator=torch.Generator().manual_seed(cfg.seed or 0))

    def train_dataloader(self):
        self.train_loader = self._loader("train")
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = self._loader("val")
        return self.val_loader

    def test_dataloader(self):
        return self._loader("test", self.test_cfg.split)

    def mini_dataloader(self):
        return self._loader("test", "mini")
