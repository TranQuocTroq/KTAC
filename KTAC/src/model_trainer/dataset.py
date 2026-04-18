"""WSI dataset and DataLoader utilities for MIL classification.

Loads pre-extracted patch feature bags (.pt or .h5) and associates them
with class-level text prompts for cross-modal training.

Example:
    >>> ds = WSIFocusDataset(config=cfg, manifest_path="data/dataset.csv",
    ...                      prompts_path="data/prompts/UBC-OCEAN_text_prompt.csv",
    ...                      num_classes=5)
    >>> ds.set_tokenizer(tokenizer)
    >>> item = ds[0]   # dict with bag_features, input_ids, attention_mask, label
"""

import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WSIFocusDataset(Dataset):
    """Multiple Instance Learning dataset for Whole Slide Image classification.

    Loads patch-level feature bags from ``.pt`` or ``.h5`` files and maps
    each slide to a class-specific text prompt. Supports lazy tokenization
    via ``set_tokenizer``.

    Args:
        config (dict, optional): Full configuration dict; used to read
            ``paths.feat_dirs`` and ``model.image_feature_dim``.
        manifest_path (str, optional): Path to the label CSV file containing
            ``slide_id`` (or ``image_id``) and ``label`` columns.
        prompts_path (str, optional): Path to a CSV file where each row
            is a text prompt for one class (row index maps to class label).
        num_classes (int): Number of output classes. Defaults to ``5``.
        feat_dirs (list[str], optional): Override feature directories.
        tokenizer: HuggingFace tokenizer; can be set later via
            ``set_tokenizer``.
    """

    # Class-level cache so multiple Subset objects share one label map
    _labels_map_cache: dict | None = None

    def __init__(
        self,
        config: dict | None = None,
        dataset_name: str | None = None,
        manifest_path: str | None = None,
        prompts_path: str | None = None,
        num_classes: int = 5,
        label_path: str | None = None,
        feat_dirs: list[str] | None = None,
        tokenizer=None,
    ) -> None:
        self.input_dim   = config["model"]["image_feature_dim"] if config else 512
        self.max_text_len = 256
        self.tokenizer   = tokenizer
        self.num_classes = num_classes

        # --- Feature directories ---
        if feat_dirs:
            self.feat_dirs = feat_dirs
        elif config and "paths" in config and "feat_dirs" in config["paths"]:
            self.feat_dirs = config["paths"]["feat_dirs"]
        else:
            self.feat_dirs = ["data/pt"]

        # --- Labels and slide IDs ---
        if label_path is None and manifest_path:
            label_path = manifest_path

        if label_path and os.path.exists(label_path):
            if WSIFocusDataset._labels_map_cache is None:
                self._load_labels(label_path)
            self.labels_map = WSIFocusDataset._labels_map_cache

            df = pd.read_csv(label_path)
            cols = df.columns.str.lower()
            if "image_id" in cols:
                id_col_idx = cols.get_loc("image_id")
            elif "slide_id" in cols:
                id_col_idx = cols.get_loc("slide_id")
            else:
                id_col_idx = 0
            self.slide_ids = (
                df.iloc[:, id_col_idx]
                .astype(str)
                .str.replace("_bag", "", regex=False)
                .tolist()
            )
        else:
            self.labels_map = {}
            self.slide_ids  = []

        # --- Text prompts (one per class, indexed by label integer) ---
        self.class_prompts   = {}
        self.default_prompt  = (
            "Microscopic histopathology image of ovarian tissue showing cancerous subtypes."
        )

        # Row mapping: class index → CSV row index
        # Matches the ordering in the two-scale UBC-OCEAN prompt file.
        _prompt_row_map = {0: 0, 1: 3, 2: 1, 3: 2, 4: 4}

        if prompts_path and os.path.exists(prompts_path):
            try:
                df_prompts  = pd.read_csv(prompts_path, header=None)
                prompt_list = df_prompts[0].astype(str).tolist()
                for label_idx, row_idx in _prompt_row_map.items():
                    if row_idx < len(prompt_list):
                        self.class_prompts[label_idx] = prompt_list[row_idx].strip()
                    else:
                        print(f"[WARNING] Prompt file missing row {row_idx} for class {label_idx}.")
                print(f"Loaded prompts from: {prompts_path}")
            except Exception as exc:
                print(f"[ERROR] Could not read prompt file: {exc}")
        else:
            print(f"[WARNING] Prompt file not found at: {prompts_path}. Using default prompt.")

    def set_tokenizer(self, tokenizer) -> None:
        """Assign a HuggingFace tokenizer for text encoding.

        Args:
            tokenizer: Pre-loaded tokenizer compatible with the text encoder.
        """
        self.tokenizer = tokenizer

    def _load_labels(self, label_path: str) -> None:
        """Build the slide_id → integer label mapping and cache it.

        Args:
            label_path (str): Path to the label CSV file.
        """
        df   = pd.read_csv(label_path)
        cols = df.columns.str.lower()

        lbl_col = df.columns[cols.get_loc("label")] if "label" in cols else df.columns[1]
        if "image_id" in cols:
            id_col = df.columns[cols.get_loc("image_id")]
        elif "slide_id" in cols:
            id_col = df.columns[cols.get_loc("slide_id")]
        else:
            id_col = df.columns[0]

        df[id_col] = df[id_col].astype(str).str.replace("_bag", "", regex=False)

        # Encode string labels to integers if necessary
        unique_labels = sorted(df[lbl_col].unique())
        if unique_labels and isinstance(unique_labels[0], str) and not str(unique_labels[0]).isdigit():
            encoder = {lbl: i for i, lbl in enumerate(unique_labels)}
            df[lbl_col] = df[lbl_col].map(encoder)

        WSIFocusDataset._labels_map_cache = dict(zip(df[id_col], df[lbl_col]))

    def _load_features(self, slide_id: str) -> torch.Tensor:
        """Load patch feature bag for one slide from .pt or .h5 files.

        Searches ``self.feat_dirs`` in order. Returns a zero tensor if no
        file is found.

        Args:
            slide_id (str): Slide identifier (without file extension).

        Returns:
            torch.Tensor: Feature tensor of shape ``[N, feat_dim]``.
        """
        for d in self.feat_dirs:
            pt_path = os.path.join(d, f"{slide_id}.pt")
            if os.path.exists(pt_path):
                try:
                    data = torch.load(pt_path, map_location="cpu")
                    if isinstance(data, dict):
                        for key in ("features", "feature", "feats"):
                            if key in data:
                                return data[key]
                        return next(v for v in data.values() if isinstance(v, torch.Tensor))
                    return data
                except Exception:
                    pass

            h5_path = os.path.join(d, f"{slide_id}.h5")
            if os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, "r") as f:
                        if "features" in f:
                            return torch.from_numpy(f["features"][:])
                except Exception:
                    pass

        return torch.zeros(1, self.input_dim)

    def __len__(self) -> int:
        """Return the number of slides in this dataset."""
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return the feature bag, tokenized prompt, and label for one slide.

        Args:
            idx (int): Slide index.

        Returns:
            dict: Keys — ``bag_features``, ``input_ids``, ``attention_mask``,
                ``label``, ``wsi_id``.
        """
        slide_id = self.slide_ids[idx]
        label    = int(self.labels_map.get(slide_id, 0))
        features = self._load_features(slide_id)

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        text_prompt = self.class_prompts.get(label, self.default_prompt)

        if self.tokenizer:
            tokenized = self.tokenizer(
                text_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            input_ids      = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
        else:
            input_ids      = torch.empty(self.max_text_len, dtype=torch.long)
            attention_mask = torch.empty(self.max_text_len, dtype=torch.long)

        return {
            "bag_features":   features.float(),
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "label":          torch.tensor(label).long(),
            "wsi_id":         slide_id,
        }


def custom_collate(batch: list) -> dict | None:
    """Collate variable-length feature bags into a padded batch.

    Pads bags to the maximum length in the batch using zero vectors and
    creates a boolean key-padding mask (``True`` = padded position).

    Args:
        batch (list[dict]): List of items returned by ``__getitem__``.

    Returns:
        dict | None: Batched tensors, or ``None`` if all items are invalid.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    labels         = torch.stack([item["label"] for item in batch])
    input_ids      = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    wsi_ids        = [item["wsi_id"] for item in batch]

    bags       = [item["bag_features"] for item in batch]
    feat_dim   = bags[0].shape[-1]
    max_len    = max(b.shape[0] for b in bags)

    padded_bags       = torch.zeros(len(batch), max_len, feat_dim)
    key_padding_mask  = torch.ones(len(batch), max_len, dtype=torch.bool)

    for i, bag in enumerate(bags):
        n = bag.shape[0]
        if n > 0:
            padded_bags[i, :n]      = bag
            key_padding_mask[i, :n] = False

    return {
        "padded_bags":       padded_bags,
        "key_padding_mask":  key_padding_mask,
        "input_ids":         input_ids,
        "attention_mask":    attention_masks,
        "label":             labels,
        "wsi_id":            wsi_ids,
    }
