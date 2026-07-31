"""Dataset reading the ImageFolder layout via persisted split indices (splits.json)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class PlantVillageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        class_names: list[str],
        transform: Callable | None = None,
    ) -> None:
        self.image_root = Path(data_root) / "plantvillage"
        with open(Path(data_root) / "splits.json", encoding="utf-8") as f:
            self.paths: list[str] = json.load(f)[split]
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        rel_path = self.paths[index]
        label = self.class_to_idx[rel_path.split("/")[0]]
        with Image.open(self.image_root / rel_path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_class_names(configs_dir: Path | None = None) -> list[str]:
    from cropguard.config import CONFIG_DIR

    path = (configs_dir or CONFIG_DIR) / "classes.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)
