"""DeepFM: Factorization Machine + Deep Neural Network.

Reference: Guo et al. (2017). DeepFM: A Factorization-Machine based Neural Network
for CTR Prediction. IJCAI 2017.

Combines the power of factorization machines for modeling feature interactions
with deep neural networks for learning high-order feature patterns.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from recllm.data.base import InteractionData
from recllm.models.base import BaseModel


class _DeepFMNetwork(nn.Module):
    """DeepFM network: FM component + Deep component sharing embeddings."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int,
        mlp_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        self.n_fields = 2  # user and item fields

        # Shared embeddings for both FM and Deep components
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # FM first-order (linear) weights
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Deep component
        deep_input_dim = self.n_fields * embed_dim
        layers: list[nn.Module] = []
        in_dim = deep_input_dim
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.deep_layers = nn.Sequential(*layers)

        # Output: FM (1) + Deep (last MLP dim) -> 1
        self.output_layer = nn.Linear(mlp_dims[-1] + 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Get shared embeddings
        user_emb = self.user_embedding(user_ids)  # (B, D)
        item_emb = self.item_embedding(item_ids)  # (B, D)

        # --- FM Component ---
        # First order: sum of biases
        fm_first = (
            self.user_bias(user_ids).squeeze(-1)
            + self.item_bias(item_ids).squeeze(-1)
            + self.global_bias
        )  # (B,)

        # Second order: user-item interaction via inner product
        fm_second = (user_emb * item_emb).sum(dim=-1)  # (B,)

        fm_out = (fm_first + fm_second).unsqueeze(-1)  # (B, 1)

        # --- Deep Component ---
        deep_input = torch.cat([user_emb, item_emb], dim=-1)  # (B, 2*D)
        deep_out = self.deep_layers(deep_input)  # (B, last_mlp_dim)

        # --- Combine ---
        combined = torch.cat([fm_out, deep_out], dim=-1)  # (B, 1 + last_mlp_dim)
        return self.output_layer(combined).squeeze(-1)  # (B,)


class DeepFM(BaseModel):
    """DeepFM: Factorization Machine + Deep Neural Network.

    Jointly trains an FM component (for explicit feature interactions)
    and a deep component (for implicit high-order interactions),
    sharing the same embedding layer.

    Args:
        embed_dim: Embedding dimension for FM and Deep components.
        mlp_dims: Hidden layer sizes for the deep component.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization.
        dropout: Dropout rate in deep layers.
        loss: Loss function — "bce" for pointwise, "bpr" for pairwise.
        device: PyTorch device.

    Example:
        >>> model = DeepFM(embed_dim=32, mlp_dims=[128, 64, 32])
        >>> model.fit(train_data, epochs=20)
        >>> recs = model.recommend(user_id=1, n=10)
    """

    def __init__(
        self,
        embed_dim: int = 32,
        mlp_dims: list[int] | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.2,
        loss: str = "bce",
        device: str = "auto",
    ):
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims or [128, 64, 32]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.loss_type = loss

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model: _DeepFMNetwork | None = None
        self._all_item_ids: np.ndarray = np.array([])
        self._user_interactions: dict[int, set[int]] = {}

    def fit(
        self,
        train_data: InteractionData,
        epochs: int = 20,
        val_data: InteractionData | None = None,
        batch_size: int = 256,
        neg_ratio: int = 4,
    ) -> Self:
        """Train DeepFM model.

        Args:
            train_data: Training interactions.
            epochs: Number of training epochs.
            val_data: Optional validation data for monitoring.
            batch_size: Training batch size.
            neg_ratio: Number of negatives per positive for BCE training.

        Returns:
            Self.
        """
        encoded_data, user_map, item_map = train_data.encode_ids()
        self._user_map = user_map
        self._item_map = item_map
        n_users = len(user_map)
        n_items = len(item_map)
        self._all_item_ids = train_data.item_ids

        arrays = encoded_data.to_numpy()
        user_ids = arrays["user_id"]
        item_ids = arrays["item_id"]

        # Build interaction lookup
        self._user_interactions = {}
        for u, i in zip(user_ids, item_ids, strict=False):
            self._user_interactions.setdefault(int(u), set()).add(int(i))

        self._model = _DeepFMNetwork(
            n_users, n_items, self.embed_dim, self.mlp_dims, self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for _epoch in trange(epochs, desc="DeepFM Training"):
            self._model.train()

            if self.loss_type == "bce":
                self._train_bce(
                    user_ids, item_ids, n_items, neg_ratio, batch_size, optimizer
                )
            else:
                self._train_bpr(
                    user_ids, item_ids, n_items, batch_size, optimizer
                )

        self._model.eval()
        return self

    def _train_bce(self, user_ids, item_ids, n_items, neg_ratio, batch_size, optimizer):
        """One epoch of pointwise BCE training with negative sampling."""
        n = len(user_ids)
        pos_users = user_ids
        pos_items = item_ids
        pos_labels = np.ones(n, dtype=np.float32)

        neg_users = np.repeat(user_ids, neg_ratio)
        neg_items = np.zeros(n * neg_ratio, dtype=user_ids.dtype)
        for idx in range(len(neg_users)):
            u = int(neg_users[idx])
            neg = np.random.randint(0, n_items)
            user_items = self._user_interactions.get(u, set())
            while neg in user_items:
                neg = np.random.randint(0, n_items)
            neg_items[idx] = neg
        neg_labels = np.zeros(n * neg_ratio, dtype=np.float32)

        all_users = np.concatenate([pos_users, neg_users])
        all_items = np.concatenate([pos_items, neg_items])
        all_labels = np.concatenate([pos_labels, neg_labels])

        indices = np.random.permutation(len(all_users))
        loss_fn = nn.BCEWithLogitsLoss()

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            u = torch.tensor(all_users[batch_idx], dtype=torch.long, device=self.device)
            i = torch.tensor(all_items[batch_idx], dtype=torch.long, device=self.device)
            y = torch.tensor(
                all_labels[batch_idx], dtype=torch.float32, device=self.device
            )

            logits = self._model(u, i)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _train_bpr(self, user_ids, item_ids, n_items, batch_size, optimizer):
        """One epoch of pairwise BPR training."""
        n = len(user_ids)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_users = user_ids[batch_idx]
            batch_pos = item_ids[batch_idx]

            batch_neg = np.zeros_like(batch_pos)
            for idx, u in enumerate(batch_users):
                neg = np.random.randint(0, n_items)
                user_items = self._user_interactions.get(int(u), set())
                while neg in user_items:
                    neg = np.random.randint(0, n_items)
                batch_neg[idx] = neg

            u = torch.tensor(batch_users, dtype=torch.long, device=self.device)
            pi = torch.tensor(batch_pos, dtype=torch.long, device=self.device)
            ni = torch.tensor(batch_neg, dtype=torch.long, device=self.device)

            pos_scores = self._model(u, pi)
            neg_scores = self._model(u, ni)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        enc_users = np.array([self._user_map.get(int(u), -1) for u in user_ids])
        enc_items = np.array([self._item_map.get(int(i), -1) for i in item_ids])

        valid_mask = (enc_users >= 0) & (enc_items >= 0)
        scores = np.zeros(len(user_ids), dtype=np.float64)

        if valid_mask.any():
            u = torch.tensor(enc_users[valid_mask], dtype=torch.long, device=self.device)
            i = torch.tensor(enc_items[valid_mask], dtype=torch.long, device=self.device)
            self._model.eval()
            scores[valid_mask] = self._model(u, i).cpu().numpy()

        return scores
