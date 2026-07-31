"""A logging backend must never be able to kill a training run.

Metrics are a convenience; checkpoints and test metrics do not depend on them. Losing hours of
GPU time to a bad API key or an unreachable W&B is a bad trade, so build_logger degrades to
CSV instead of raising.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytorch_lightning")

from pytorch_lightning.loggers import CSVLogger  # noqa: E402

from cropguard.training.train import build_logger  # noqa: E402

CFG = {"wandb": {"project": "cropguard-mlops"}}


def test_disabled_mode_uses_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)

    assert isinstance(build_logger(CFG, "exp"), CSVLogger)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("entity someone not found during upsertBucket"),
        ConnectionError("network unreachable"),
        ValueError("api_key not configured"),
    ],
)
def test_wandb_failure_falls_back_instead_of_raising(monkeypatch, tmp_path, exc):
    """Whatever W&B throws - auth, entity, network - training must still start."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.chdir(tmp_path)

    import pytorch_lightning.loggers as pl_loggers

    class ExplodingWandbLogger:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def experiment(self):
            raise exc

    monkeypatch.setattr(pl_loggers, "WandbLogger", ExplodingWandbLogger, raising=False)

    logger = build_logger(CFG, "exp")
    assert isinstance(logger, CSVLogger), "a W&B failure must not propagate"


def test_entity_errors_surface_an_actionable_hint(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.chdir(tmp_path)

    import pytorch_lightning.loggers as pl_loggers

    class ExplodingWandbLogger:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def experiment(self):
            raise RuntimeError("entity abhinav not found during upsertBucket")

    monkeypatch.setattr(pl_loggers, "WandbLogger", ExplodingWandbLogger, raising=False)

    build_logger(CFG, "exp")
    assert "WANDB_ENTITY" in capsys.readouterr().out


def test_working_wandb_is_used(monkeypatch, tmp_path):
    """The fallback must not swallow a perfectly good W&B logger."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.chdir(tmp_path)

    import pytorch_lightning.loggers as pl_loggers

    class WorkingWandbLogger:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        @property
        def experiment(self):
            return object()

    monkeypatch.setattr(pl_loggers, "WandbLogger", WorkingWandbLogger, raising=False)

    logger = build_logger(CFG, "exp")
    assert isinstance(logger, WorkingWandbLogger)
    assert logger.kwargs["project"] == "cropguard-mlops"
