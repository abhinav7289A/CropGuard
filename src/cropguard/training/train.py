"""Train a CropGuard model from a YAML config.

Usage (Lightning AI studio / Colab / local):
    python -m cropguard.training.train --config configs/resnet50_baseline.yaml
    python -m cropguard.training.train --config configs/convnext_tiny.yaml --resume path/to.ckpt

Checkpoints go to checkpoints/<experiment_name>/ and training is resumable with --resume.
Set WANDB_MODE=offline to log locally, or WANDB_MODE=disabled (or simply do not install
wandb) to fall back to a CSV logger — useful on CPU-only CI runners.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, Logger

from cropguard.config import load_config
from cropguard.training.datamodule import PlantVillageDataModule
from cropguard.training.module import CropGuardModule


def build_logger(cfg: dict, experiment: str) -> Logger:
    """W&B when it is installed and enabled, otherwise a local CSV logger.

    Keeping this a soft dependency means CI and offline machines can run the exact same
    training entrypoint as the GPU box, instead of a diverging code path.
    """
    if os.environ.get("WANDB_MODE") != "disabled":
        try:
            from pytorch_lightning.loggers import WandbLogger

            return WandbLogger(project=cfg["wandb"]["project"], name=experiment)
        except (ImportError, ModuleNotFoundError):
            print("wandb not available — falling back to CSVLogger (logs/).")
    return CSVLogger("logs", name=experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--fast-dev-run", action="store_true", help="1-batch smoke test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg["seed"], workers=True)

    datamodule = PlantVillageDataModule(cfg)
    model = CropGuardModule(cfg, num_classes=datamodule.num_classes)

    experiment = cfg["experiment_name"]
    checkpoint_dir = Path("checkpoints") / experiment
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch}-{val_f1_macro:.4f}",
            monitor="val_f1_macro",
            mode="max",
            save_top_k=1,
            save_last=True,  # enables resume after Colab/Lightning disconnects
        ),
        EarlyStopping(
            monitor="val_f1_macro",
            mode="max",
            patience=cfg["train"]["early_stopping_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = build_logger(cfg, experiment)
    logger.log_hyperparams(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg["train"]["epochs"],
        precision=cfg["train"]["precision"],
        accelerator="auto",
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    if not args.fast_dev_run:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
