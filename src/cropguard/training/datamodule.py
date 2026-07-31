from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cropguard.config import data_root
from cropguard.data.dataset import PlantVillageDataset, load_class_names
from cropguard.data.transforms import eval_transforms, train_transforms


class PlantVillageDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.root = data_root(cfg)
        self.class_names = load_class_names()
        self.num_classes = len(self.class_names)

    def setup(self, stage: str | None = None) -> None:
        image_size = self.cfg["data"]["image_size"]
        self.train_set = PlantVillageDataset(
            self.root,
            "train",
            self.class_names,
            train_transforms(self.cfg["augmentation"], image_size),
        )
        self.val_set = PlantVillageDataset(
            self.root, "val", self.class_names, eval_transforms(image_size)
        )
        self.test_set = PlantVillageDataset(
            self.root, "test", self.class_names, eval_transforms(image_size)
        )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.cfg["data"]["num_workers"],
            pin_memory=True,
            persistent_workers=self.cfg["data"]["num_workers"] > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False)
