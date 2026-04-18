"""Main training entry point for KTAC.

Runs stratified k-fold cross-validation across multiple few-shot settings
(4-shot, 8-shot, 16-shot) as defined in ``configs/model_config.yaml``.
Saves the best checkpoint per fold and prints a final summary table.

Usage:
    python -m src.model_trainer.main_train --config configs/model_config.yaml
"""

import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .architecture import FocusOnSpark
from .dataset import WSIFocusDataset, custom_collate
from .engine import evaluate, train_one_epoch
from .utils import load_config, set_seed

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def clean_id(raw_id) -> str:
    """Normalize a slide ID by stripping extensions and suffixes.

    Args:
        raw_id: Raw slide identifier (string or number).

    Returns:
        str: Cleaned slide ID.
    """
    s = str(raw_id).strip()
    for suffix in (".png", ".h5", ".pt", "_bag", ".tif", ".tiff", ".csv"):
        s = s.replace(suffix, "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def get_split_indices(split_file: str, master_slide_ids: list, column: str) -> list[int]:
    """Return dataset indices for a given split column in a fold CSV.

    Args:
        split_file (str): Path to the fold split CSV.
        master_slide_ids (list): Ordered slide IDs from the master dataset.
        column (str): Column name to read (``"train"``, ``"val"``, or ``"test"``).

    Returns:
        list[int]: Indices into ``master_slide_ids`` for this split.
    """
    if not os.path.exists(split_file):
        return []
    try:
        df   = pd.read_csv(split_file)
        cols = {c.lower(): c for c in df.columns}
        col  = cols.get(column.lower())
        if col is None:
            return []
        split_ids  = set(clean_id(sid) for sid in df[col].dropna().tolist())
        master_map = {clean_id(sid): idx for idx, sid in enumerate(master_slide_ids)}
        return [master_map[sid] for sid in split_ids if sid in master_map]
    except Exception:
        return []


def find_split_file(base_dir: str, setting: str, fold: int) -> str | None:
    """Search recursively for the split CSV for a given setting and fold.

    Args:
        base_dir (str): Root splits directory.
        setting (str): Shot setting string (e.g. ``"4shot"``).
        fold (int): Fold number.

    Returns:
        str | None: Path to the split CSV, or ``None`` if not found.
    """
    candidates  = [f"split{fold}.csv", f"splits_{fold}.csv", f"split_{fold}.csv"]
    setting_key = setting.replace("shot", "").replace("s", "")

    for root, _, files in os.walk(base_dir):
        folder = os.path.basename(root)
        if setting_key in folder and "shot" in folder:
            for fname in candidates:
                if fname in files:
                    return os.path.join(root, fname)
    return None


def run_fold(
    fold: int,
    setting: str,
    config: dict,
    master_dataset: WSIFocusDataset,
) -> tuple[float, float, float]:
    """Train and evaluate the model for one fold.

    Args:
        fold (int): Fold index.
        setting (str): Few-shot setting (e.g. ``"4shot"``).
        config (dict): Full configuration dictionary.
        master_dataset (WSIFocusDataset): Full dataset; split via indices.

    Returns:
        tuple[float, float, float]: ``(test_auc, test_f1, test_accuracy)``.
            All zero if the fold split file is not found.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = find_split_file(config["paths"]["splits_dir"], setting, fold)
    if split_path is None:
        print(f"  [SKIP] Split file not found for fold {fold}, setting {setting}")
        return 0.0, 0.0, 0.0

    train_idx = get_split_indices(split_path, master_dataset.slide_ids, "train")
    val_idx   = get_split_indices(split_path, master_dataset.slide_ids, "val")
    test_idx  = get_split_indices(split_path, master_dataset.slide_ids, "test")

    if not train_idx:
        return 0.0, 0.0, 0.0

    batch_size  = config["training"]["batch_size"]
    num_workers = config.get("num_workers", 0)

    train_loader = DataLoader(Subset(master_dataset, train_idx), batch_size=batch_size, shuffle=True,  collate_fn=custom_collate, num_workers=num_workers)
    val_loader   = DataLoader(Subset(master_dataset, val_idx),   batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)
    test_loader  = DataLoader(Subset(master_dataset, test_idx),  batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)

    local_cfg = {**config, "model": {**config["model"], "num_classes": master_dataset.num_classes}}
    model     = FocusOnSpark(local_cfg).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.0))

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{config['run_name']}_{setting}_fold{fold}.pt")

    best_val_loss = float("inf")
    patience      = config["training"].get("early_stopping_patience", 15)
    no_improve    = 0
    start_patience_epoch = 40

    print(f"\n--- Fold {fold} | {setting} ---")

    for epoch in range(config["training"]["epochs"]):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate(model, val_loader, criterion, device, master_dataset.num_classes)
        scheduler.step()

        print(
            f"  Ep {epoch + 1:02d} | "
            f"train_loss: {train_loss:.3f} | "
            f"val_loss: {val_metrics['loss']:.3f} | "
            f"val_acc: {val_metrics['accuracy']:.3f} | "
            f"val_f1: {val_metrics['f1_score']:.3f} | "
            f"val_auc: {val_metrics['auc']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            state = model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict()
            torch.save(state, ckpt_path)
            no_improve = 0
        else:
            if epoch >= start_patience_epoch:
                no_improve += 1
                if no_improve >= patience:
                    print("  Early stopping.")
                    break
            else:
                no_improve = 0

    if not os.path.exists(ckpt_path):
        return 0.0, 0.0, 0.0

    best_model = FocusOnSpark(local_cfg).to(device)
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(best_model, test_loader, criterion, device, master_dataset.num_classes)

    print(
        f"  TEST → auc: {test_metrics['auc']:.4f} | "
        f"f1: {test_metrics['f1_score']:.4f} | "
        f"acc: {test_metrics['accuracy']:.4f}"
    )
    return test_metrics["auc"], test_metrics["f1_score"], test_metrics["accuracy"]


def main() -> None:
    """Entry point: run all experiments defined in the config file."""
    parser = argparse.ArgumentParser(description="KTAC training script.")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    os.makedirs("checkpoints", exist_ok=True)

    tokenizer     = AutoTokenizer.from_pretrained(config["model"]["text_encoder_name"])
    summary_rows: list[dict] = []

    for exp in config["experiments_to_run"]:
        WSIFocusDataset._labels_map_cache = None  # Reset cache between experiments
        master_ds = WSIFocusDataset(
            config=config,
            dataset_name=exp["dataset_name"],
            manifest_path=exp["manifest_file"],
            prompts_path=exp["prompts_file"],
            num_classes=exp["num_classes"],
        )
        master_ds.set_tokenizer(tokenizer)

        for setting in exp["settings"]:
            aucs, f1s, accs = [], [], []
            print(f"\n{'=' * 60}\n  {exp['dataset_name']} / {setting}\n{'=' * 60}")

            for fold in range(10):
                auc, f1, acc = run_fold(fold, setting, config, master_ds)
                if auc > 0.0:
                    aucs.append(auc)
                    f1s.append(f1)
                    accs.append(acc)
                torch.cuda.empty_cache()

            if aucs:
                summary_rows.append({
                    "Setting": setting,
                    "AUC":     f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
                    "F1":      f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
                    "ACC":     f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
                })

    print("\n" + "=" * 60)
    print("  Final Results")
    print("=" * 60)
    print(pd.DataFrame(summary_rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
