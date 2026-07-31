"""Shared fixtures: a tiny synthetic ImageFolder dataset, so tests never need PlantVillage."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

CLASSES = ["Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight"]


def write_image(path: Path, size: tuple[int, int] = (256, 256), color: str = "green") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, quality=95)
    return path


@pytest.fixture
def image_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """<root>/plantvillage/<class>/<n>.jpg with 6 valid images per class."""
    root = tmp_path / "plantvillage"
    for class_index, class_name in enumerate(CLASSES):
        for i in range(6):
            write_image(
                root / class_name / f"{i:03d}.jpg", color=("green", "olive", "lime")[class_index]
            )
    return root
