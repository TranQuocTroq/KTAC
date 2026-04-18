"""Core compression and aggregation modules for KTAC.

Implements the three-stage visual token compression pipeline from FOCUS
and the cross-modal aggregation head used in FocusOnSpark.

Modules:
    - ``FocusModules``       — global patch filtering + sequential token compression
    - ``CrossModalAggregator`` — multi-head cross-attention between text and image tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocusModules(nn.Module):
    """Visual token compression utilities.

    Implements two of the three FOCUS compression stages:

    - **Stage 1** — ``compute_patch_similarity``: global redundancy removal
      via sliding-window cosine similarity thresholding.
    - **Stage 3** — ``spatial_token_compression``: sequential local redundancy
      removal by comparing consecutive token pairs.

    Stage 2 (language-guided selection) is handled inside ``FocusOnSpark.forward``.

    Args:
        window_size (int): Number of patches per sliding window for global
            filtering. Defaults to ``32``.
        sim_threshold (float): Cosine similarity threshold for sequential
            compression; tokens with similarity above this value are merged.
            Defaults to ``0.7``.
    """

    def __init__(self, window_size: int = 32, sim_threshold: float = 0.7) -> None:
        super().__init__()
        self.window_size   = window_size
        self.sim_threshold = sim_threshold

    def compute_patch_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 1: Remove globally redundant patches via sliding window.

        Within each window, patches whose mean cosine similarity to their
        neighbours exceeds a dynamic threshold (μ + σ) are considered
        redundant and removed.

        Args:
            x (torch.Tensor): Patch feature tensor of shape ``[N, D]``.

        Returns:
            torch.Tensor: 1-D index tensor of retained patch positions.
        """
        N, _ = x.shape
        if N < self.window_size:
            return torch.arange(N, device=x.device)

        x_norm = F.normalize(x, p=2, dim=-1)
        kept   = []

        for start in range(0, N, self.window_size):
            window = x_norm[start : start + self.window_size]
            if len(window) < 2:
                kept.append(torch.arange(start, min(start + self.window_size, N), device=x.device))
                continue

            sim_matrix = torch.matmul(window, window.T)          # [W, W]
            mean_sim   = sim_matrix.mean(dim=1)                  # [W]
            threshold  = mean_sim.mean() + mean_sim.std()

            keep_mask  = ~(mean_sim > threshold)
            indices    = torch.where(keep_mask)[0] + start

            # Guarantee at least one patch per window
            if indices.numel() == 0:
                indices = torch.tensor([mean_sim.argmin().item() + start], device=x.device)

            kept.append(indices)

        return torch.cat(kept) if kept else torch.arange(N, device=x.device)

    def spatial_token_compression(self, features: torch.Tensor) -> torch.Tensor:
        """Stage 3: Remove locally redundant consecutive tokens.

        Computes cosine similarity between every adjacent pair of tokens.
        Tokens whose similarity with both neighbours exceeds
        ``sim_threshold`` are discarded, preserving spatial coherence.

        Args:
            features (torch.Tensor): Selected token tensor of shape ``[N, D]``.

        Returns:
            torch.Tensor: Compressed token tensor of shape ``[M, D]``
                where ``M ≤ N``.
        """
        N, _ = features.shape
        if N <= 1:
            return features

        norm      = F.normalize(features, p=2, dim=-1)
        sim       = F.cosine_similarity(norm[:-1], norm[1:], dim=-1)  # [N-1]
        keep_mask = sim < self.sim_threshold                           # True = keep

        # Always keep the first token; then keep tokens where similarity drops
        kept_indices = torch.cat([
            torch.tensor([0], device=features.device),
            torch.where(keep_mask)[0] + 1,
        ])
        return features[kept_indices]


class CrossModalAggregator(nn.Module):
    """Multi-head cross-attention aggregator for text-guided image features.

    Uses text embeddings as queries and compressed visual tokens as keys/values,
    producing a text-attended slide-level representation.

    Args:
        embed_dim (int): Common embedding dimension for Q, K, V projections.
        num_heads (int): Number of attention heads. Defaults to ``8``.
        dropout (float): Dropout probability on output projection.
            Defaults to ``0.3``.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim  = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-head cross-attention.

        Args:
            query (torch.Tensor): Text embeddings of shape ``[B, T, D]``.
            key (torch.Tensor): Visual tokens of shape ``[B, N, D]``.
            value (torch.Tensor): Visual tokens of shape ``[B, N, D]``.

        Returns:
            torch.Tensor: Attended output of shape ``[B, T, D]``.
        """
        B, T, _ = query.shape
        _, N, _ = key.shape

        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn   = F.scaled_dot_product_attention(q, k, v)
        attn   = attn.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        output = self.o_proj(attn)
        return output
