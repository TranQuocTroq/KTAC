"""Training and evaluation loop functions.

Example:
    >>> train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, device)
    >>> metrics = evaluate(model, val_loader, criterion, device, num_classes=5)
    >>> print(metrics["auc"], metrics["f1_score"])
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model (nn.Module): Model to train.
        dataloader: DataLoader yielding batches from ``custom_collate``.
        optimizer (Optimizer): Gradient optimizer.
        criterion (nn.Module): Loss function (e.g. ``CrossEntropyLoss``).
        device (torch.device): Target device.

    Returns:
        tuple[float, float]: ``(mean_loss, accuracy)`` over the epoch.
    """
    model.train()
    total_loss  = 0.0
    all_labels: list = []
    all_preds:  list = []

    for batch in dataloader:
        if batch is None:
            continue

        input_ids        = batch["input_ids"].to(device)
        attention_mask   = batch["attention_mask"].to(device)
        labels           = batch["label"].to(device)
        padded_bags      = batch["padded_bags"].to(device)
        key_padding_mask = batch["key_padding_mask"].to(device)

        optimizer.zero_grad()
        logits = model(padded_bags, key_padding_mask, input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    mean_loss = total_loss / max(len(dataloader), 1)
    accuracy  = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return mean_loss, accuracy


def evaluate(
    model: torch.nn.Module,
    dataloader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    """Evaluate the model on a validation or test split.

    Computes loss, accuracy, weighted F1, and macro OvR AUC.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader: DataLoader yielding batches from ``custom_collate``.
        criterion (nn.Module): Loss function.
        device (torch.device): Target device.
        num_classes (int): Number of output classes (used for AUC computation).

    Returns:
        dict[str, float]: Keys — ``"loss"``, ``"accuracy"``, ``"f1_score"``,
            ``"auc"``.
    """
    model.eval()
    total_loss  = 0.0
    all_labels: list = []
    all_preds:  list = []
    all_probs:  list = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            input_ids        = batch["input_ids"].to(device)
            attention_mask   = batch["attention_mask"].to(device)
            labels           = batch["label"].to(device)
            padded_bags      = batch["padded_bags"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)

            logits = model(padded_bags, key_padding_mask, input_ids, attention_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if not all_labels:
        return {"loss": 0.0, "accuracy": 0.0, "f1_score": 0.0, "auc": 0.5}

    mean_loss = total_loss / max(len(dataloader), 1)
    accuracy  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    auc = 0.5
    try:
        if len(np.unique(all_labels)) >= 2:
            if num_classes == 2:
                auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
            else:
                auc = roc_auc_score(
                    all_labels, all_probs,
                    multi_class="ovr", labels=list(range(num_classes)),
                )
    except Exception:
        pass

    return {"loss": mean_loss, "accuracy": accuracy, "f1_score": f1, "auc": auc}
