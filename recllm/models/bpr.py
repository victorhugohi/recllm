"""Bayesian Personalized Ranking (BPR) model.

Reference: Rendle et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback.
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


class BPR(BaseModel):
    """Bayesian Personalized Ranking with matrix factorization.

    Learns user and item embeddings optimized for pairwise ranking:
    for each user, observed items should be ranked higher than
    unobserved items.

    Args:
        embedding_dim: Dimensionality of user/item embeddings.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization weight.
        device: PyTorch device ("cpu", "cuda", "auto").

    Example:
        >>> model = BPR(embedding_dim=64)
        >>> model.fit(train_data, epochs=50)
        >>> recs = model.recommend(user_id=1, n=10)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
    ):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._user_emb: nn.Embedding | None = None
        self._item_emb: nn.Embedding | None = None
        self._all_item_ids: np.ndarray = np.array([])
        self._n_users: int = 0
        self._n_items: int = 0
        self._user_interactions: dict[int, set[int]] = {}

    def fit(
        self,
        train_data: InteractionData,
        epochs: int = 50,
        val_data: InteractionData | None = None,
        batch_size: int = 1024,
        n_negatives: int = 1,
    ) -> Self:
        """Train BPR model with pairwise ranking loss.

        Args:
            train_data: Training interactions.
            epochs: Number of training epochs.
            val_data: Optional validation data (for logging, no early stopping yet).
            batch_size: Training batch size.
            n_negatives: Number of negative samples per positive pair.

        Returns:
            Self.
        """
        # Encode IDs to contiguous range
        encoded_data, user_map, item_map = train_data.encode_ids()
        self._user_map = user_map
        self._item_map = item_map
        self._reverse_item_map = {v: k for k, v in item_map.items()}

        self._n_users = len(user_map)
        self._n_items = len(item_map)
        self._all_item_ids = train_data.item_ids

        # Build user -> items mapping for negative sampling
        arrays = encoded_data.to_numpy()
        user_ids = arrays["user_id"]
        item_ids = arrays["item_id"]

        self._user_interactions = {}
        for u, i in zip(user_ids, item_ids, strict=False):
            self._user_interactions.setdefault(int(u), set()).add(int(i))

        # Initialize embeddings
        self._user_emb = nn.Embedding(self._n_users, self.embedding_dim).to(self.device)
        self._item_emb = nn.Embedding(self._n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_normal_(self._user_emb.weight)
        nn.init.xavier_normal_(self._item_emb.weight)

        optimizer = torch.optim.Adam(
            list(self._user_emb.parameters()) + list(self._item_emb.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        n_interactions = len(user_ids)
        indices = np.arange(n_interactions)

        for _epoch in trange(epochs, desc="BPR Training"):
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_interactions, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_users = user_ids[batch_idx]
                batch_pos_items = item_ids[batch_idx]

                # Sample negative items
                batch_neg_items = np.zeros_like(batch_pos_items)
                for i, u in enumerate(batch_users):
                    neg = np.random.randint(0, self._n_items)
                    user_items = self._user_interactions.get(int(u), set())
                    while neg in user_items:
                        neg = np.random.randint(0, self._n_items)
                    batch_neg_items[i] = neg

                # Forward pass
                u = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                pi = torch.tensor(batch_pos_items, dtype=torch.long, device=self.device)
                ni = torch.tensor(batch_neg_items, dtype=torch.long, device=self.device)

                u_emb = self._user_emb(u)
                pi_emb = self._item_emb(pi)
                ni_emb = self._item_emb(ni)

                pos_scores = (u_emb * pi_emb).sum(dim=1)
                neg_scores = (u_emb * ni_emb).sum(dim=1)

                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return self

    @torch.no_grad()
    def predict(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        """Predict scores for user-item pairs.

        Args:
            user_ids: Array of original user IDs.
            item_ids: Array of original item IDs.

        Returns:
            Array of predicted scores.
        """
        if self._user_emb is None or self._item_emb is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        # Map to encoded IDs
        enc_users = np.array([self._user_map.get(int(u), -1) for u in user_ids])
        enc_items = np.array([self._item_map.get(int(i), -1) for i in item_ids])

        # Handle unknown users/items
        valid_mask = (enc_users >= 0) & (enc_items >= 0)
        scores = np.zeros(len(user_ids), dtype=np.float64)

        if valid_mask.any():
            u = torch.tensor(enc_users[valid_mask], dtype=torch.long, device=self.device)
            i = torch.tensor(enc_items[valid_mask], dtype=torch.long, device=self.device)

            u_emb = self._user_emb(u)
            i_emb = self._item_emb(i)
            scores[valid_mask] = (u_emb * i_emb).sum(dim=1).cpu().numpy()

        return scores
