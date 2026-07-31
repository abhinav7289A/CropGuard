from __future__ import annotations

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score


class CropGuardModule(pl.LightningModule):
    """timm classifier wrapper. Architecture, LR, and augmentation all come from cfg,
    so the same module trains the ResNet50 baseline and the ConvNeXt-Tiny challenger."""

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters("cfg", "num_classes")
        self.cfg = cfg
        self.model = timm.create_model(
            cfg["model"]["name"],
            pretrained=cfg["model"]["pretrained"],
            num_classes=num_classes,
            drop_rate=cfg["model"].get("dropout", 0.0),
        )
        self.label_smoothing = cfg["train"].get("label_smoothing", 0.0)

        metric_args = {"task": "multiclass", "num_classes": num_classes}
        self.train_acc = Accuracy(**metric_args)
        self.val_acc = Accuracy(**metric_args)
        self.val_f1 = F1Score(average="macro", **metric_args)
        self.test_acc = Accuracy(**metric_args)
        self.test_f1 = F1Score(average="macro", **metric_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1_macro", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.log("test_acc", self.test_acc)
        self.log("test_f1_macro", self.test_f1)

    def configure_optimizers(self):
        tcfg = self.cfg["train"]
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(tcfg["epochs"] - tcfg.get("warmup_epochs", 0), 1)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
