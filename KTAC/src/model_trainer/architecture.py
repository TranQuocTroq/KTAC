"""FocusOnSpark: three-stage visual token compression MIL classifier.

Implements the KTAC model architecture combining:
    - Frozen DistilBERT text encoder for class-level semantic anchors
    - Three-stage visual token compression pipeline (FOCUS-inspired)
    - Cross-modal aggregation for text-guided slide-level prediction

Example:
    >>> model = FocusOnSpark(config)
    >>> logits = model(padded_bags, key_padding_mask, input_ids, attention_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .modules import CrossModalAggregator, FocusModules


class FocusOnSpark(nn.Module):
    """Three-stage visual compression MIL model for WSI classification.

    Architecture:
        1. **Text Encoder** — frozen DistilBERT; CLS token as class anchor.
        2. **Feature Encoder** — linear projection + LayerNorm + ReLU + Dropout.
        3. **Stage 1** — Global patch redundancy removal (sliding-window similarity).
        4. **Stage 2** — Language-guided token selection (top-k by text relevance).
        5. **Stage 3** — Sequential visual token compression (SVTC).
        6. **Aggregator** — Cross-modal multi-head attention (text queries, image keys/values).
        7. **Classifier** — Linear head over aggregated text-attended representation.

    Args:
        config (dict): Configuration dictionary. Expected keys under ``"model"``:
            - ``image_feature_dim`` (int): Input feature dimension.
            - ``projection_dim`` (int): Internal projection dimension.
            - ``compression_ratio`` (float): Fraction of tokens kept in Stage 2.
            - ``text_encoder_name`` (str): HuggingFace model name for the text encoder.
            - ``num_classes`` (int): Number of output classes.
            - ``aggregator.num_heads`` (int): Attention heads in the aggregator.
            - ``aggregator.dropout`` (float): Dropout in the aggregator.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        cfg = config.get("model", config)

        # --- Text Encoder (frozen) ---
        self.text_encoder = AutoModel.from_pretrained(cfg["text_encoder_name"])
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        feat_dim  = cfg["image_feature_dim"]
        proj_dim  = cfg["projection_dim"]
        num_heads = cfg.get("aggregator", {}).get("num_heads", 8)
        agg_drop  = cfg.get("aggregator", {}).get("dropout", 0.3)

        # --- Feature Encoder ---
        self.feature_encoder = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Project text CLS token into shared embedding space
        self.text_proj = nn.Linear(768, proj_dim)

        # --- Compression hyperparameters ---
        self.gamma = cfg.get("compression_ratio", 0.8)  # Stage 2 keep ratio
        self.L_max = 1024                                 # Hard cap on token count

        # --- Compression modules ---
        self.helpers = FocusModules(window_size=32, sim_threshold=0.7)

        # --- Aggregation and classification ---
        self.aggregator = CrossModalAggregator(embed_dim=proj_dim, num_heads=num_heads, dropout=agg_drop)
        self.classifier = nn.Linear(proj_dim, cfg["num_classes"])

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier-uniform initialization for linear layers.

        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        padded_bags: torch.Tensor,
        key_padding_mask: torch.BoolTensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the full KTAC pipeline.

        Args:
            padded_bags (torch.Tensor): Padded patch feature bags of shape
                ``[B, N_max, feat_dim]``.
            key_padding_mask (torch.BoolTensor): Boolean mask of shape
                ``[B, N_max]`` where ``True`` marks padded positions.
            input_ids (torch.Tensor): Tokenized text IDs of shape
                ``[B, seq_len]``.
            attention_mask (torch.Tensor): Text attention mask of shape
                ``[B, seq_len]``.

        Returns:
            torch.Tensor: Class logits of shape ``[B, num_classes]``.
        """
        # --- A. Text encoding ---
        with torch.no_grad():
            text_out  = self.text_encoder(input_ids, attention_mask)
            text_feat = text_out.last_hidden_state[:, 0, :]   # CLS token: [B, 768]
        text_emb = self.text_proj(text_feat).unsqueeze(1)     # [B, 1, D]

        # --- B. Image encoding ---
        img_emb = self.feature_encoder(padded_bags.float())   # [B, N, D]

        B, N, D = img_emb.shape

        # --- C. Three-stage token compression (per sample) ---
        compressed_bags = []
        for b in range(B):
            valid_len = torch.sum(~key_padding_mask[b]).long()
            feat = img_emb[b, :valid_len]                     # [N_valid, D]

            if feat.shape[0] == 0:
                compressed_bags.append(img_emb.new_zeros((1, D)))
                continue

            # Stage 1: Global redundancy removal
            keep_idx = self.helpers.compute_patch_similarity(feat)
            feat = feat[keep_idx]

            if feat.shape[0] == 0:
                compressed_bags.append(img_emb[b, 0].unsqueeze(0))
                continue

            # Stage 2: Language-guided token selection
            # Relevance = cosine similarity between each token and text anchor
            relevance = torch.matmul(feat, text_emb[b].T).squeeze(-1)  # [N']
            k = min(self.L_max, max(1, int(self.gamma * feat.shape[0])))
            _, topk_idx = torch.topk(relevance, k=k, dim=0)
            topk_idx, _ = torch.sort(topk_idx)                # Preserve spatial order
            feat = feat[topk_idx]

            # Stage 3: Sequential visual token compression (SVTC)
            feat = self.helpers.spatial_token_compression(feat)
            if feat.shape[0] == 0:
                feat = img_emb[b, 0].unsqueeze(0)

            compressed_bags.append(feat)

        # --- D. Re-pad compressed bags ---
        max_len     = max(max(f.shape[0] for f in compressed_bags), 1)
        new_img_emb = img_emb.new_zeros(B, max_len, D)
        for b, feat in enumerate(compressed_bags):
            new_img_emb[b, : feat.shape[0]] = feat

        # --- E. Cross-modal aggregation and classification ---
        attended = self.aggregator(text_emb, new_img_emb, new_img_emb)  # [B, 1, D]
        logits   = self.classifier(attended.squeeze(1))                  # [B, C]
        return logits
