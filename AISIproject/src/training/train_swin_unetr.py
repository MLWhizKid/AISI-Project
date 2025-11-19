from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism

from src.data.polyp_datamodule import create_dataloaders

LOGGER = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any], device: torch.device) -> SwinUNETR:
    model_cfg = cfg["model"]
    model = SwinUNETR(
        img_size=cfg["data"]["resize"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        feature_size=model_cfg.get("feature_size", 48),
        use_checkpoint=True,
    )
    model = model.to(device)
    return model


def train(cfg: Dict[str, Any]) -> None:
    set_determinism(seed=cfg.get("seed", 42))
    device = torch.device(cfg.get("system", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Using device: %s", device)

    train_loader, val_loader = create_dataloaders(
        train_dir=cfg["data"]["train_dir"],
        val_dir=cfg["data"]["val_dir"],
        image_suffix=cfg["data"].get("image_suffix", ".jpg"),
        mask_suffix=cfg["data"].get("mask_suffix", ".png"),
        image_size=tuple(cfg["data"]["resize"]),
        cache_rate=cfg["data"].get("cache_rate", 0.1),
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
    )

    model = build_model(cfg, device)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True) and device.type == "cuda")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    best_dice = 0.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            scaler.scale(loss).backward()
            if cfg["training"].get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if step % 10 == 0:
                LOGGER.info("Epoch %d step %d loss %.4f", epoch, step, loss.item())

        LOGGER.info("Epoch %d train loss %.4f", epoch, epoch_loss / max(1, len(train_loader)))

        if epoch % cfg["training"]["val_every"] == 0:
            dice = validate(model, val_loader, dice_metric, post_pred, device)
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(model, cfg["project"], epoch, best_dice)
            LOGGER.info("Epoch %d val dice %.4f (best %.4f)", epoch, dice, best_dice)


def validate(model, val_loader, dice_metric, post_pred, device):
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            preds = post_pred(logits)
            dice_metric(preds, masks)
    dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return dice


def save_checkpoint(model: torch.nn.Module, project: str, epoch: int, metric: float):
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = out_dir / f"{project}_swin_unetr_epoch{epoch}_dice{metric:.4f}.pt"
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "dice": metric}, ckpt_path)
    LOGGER.info("Saved checkpoint to %s", ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin-UNETR baseline for polyp segmentation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_swin_unetr.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)
