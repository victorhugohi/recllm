"""LightGCN: Simplifying and Powering Graph Convolution Network.

Reference: He et al. (2020). LightGCN: Simplifying and Powering Graph
Convolution Network for Recommendation. SIGIR 2020.
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


class LightGCN(BaseModel):
    """LightGCN for collaborative filtering.

    Learns user and item embeddings by propagating them on the
    user-item interaction graph. Uses simplified graph convolution
    (no feature transformation or nonlinear activation) and
    layer-combination via mean pooling.

    Args:
        embedding_dim: Embedding dimension.
        n_layers: Number of graph convolution layers.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization (applied to embeddings).
        device: PyTorch device.

    Example:
        >>> model = LightGCN(embedding_dim=64, n_layers=3)
        >>> model.fit(train_data, epochs=50)
        >>> recs = model.recommend(user_id=1, n=10)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_layers: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._user_emb: nn.Embedding | None = None
        self._item_emb: nn.Embedding | None = None
        self._adj_matrix: torch.Tensor | None = None
        self._all_item_ids: np.ndarray = np.array([])
        self._n_users: int = 0
        self._n_items: int = 0
        self._user_interactions: dict[int, set[int]] = {}

    def _build_adj_matrix(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> torch.Tensor:
        """Build normalized adjacency matrix for bipartite user-item graph.

        The adjacency matrix A of the bipartite graph is:
        [[0, R], [R^T, 0]] where R is the user-item interaction matrix.
        Normalized: D^{-1/2} A D^{-1/2}.
        """
        n = self._n_users + self._n_items

        # Build edge index: user -> item (shifted by n_users) and reverse
        rows = np.concatenate([user_ids, item_ids + self._n_users])
        cols = np.concatenate([item_ids + self._n_users, user_ids])
        values = np.ones(len(rows), dtype=np.float32)

        # Compute degree for normalization
        indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        adj = torch.sparse_coo_tensor(
            indices,
            torch.tensor(values),
            size=(n, n),
        ).coalesce()

        # D^{-1/2}
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        degree_inv_sqrt = torch.where(
            degree > 0, degree.pow(-0.5), torch.zeros_like(degree)
        )

        # Normalize: D^{-1/2} A D^{-1/2}
        d_left = degree_inv_sqrt[rows]
        d_right = degree_inv_sqrt[cols]
        norm_values = torch.tensor(values) * d_left * d_right

        norm_adj = torch.sparse_coo_tensor(
            indices,
            norm_values,
            size=(n, n),
        ).coalesce()

        return norm_adj.to(self.device)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform LightGCN layer propagation.

        Returns:
            (user_final_emb, item_final_emb) after layer-combination.
        """
        all_emb = torch.cat([self._user_emb.weight, self._item_emb.weight], dim=0)
        layer_embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self._adj_matrix, all_emb)
            layer_embs.append(all_emb)

        # Mean pooling across layers (including layer 0)
        final_emb = torch.stack(layer_embs, dim=0).mean(dim=0)

        user_emb = final_emb[: self._n_users]
        item_emb = final_emb[self._n_users :]
        return user_emb, item_emb

    def fit(
        self,
        train_data: InteractionData,
        epochs: int = 50,
        val_data: InteractionData | None = None,
        batch_size: int = 2048,
    ) -> Self:
        """Train LightGCN with BPR loss.

        Args:
            train_data: Training interactions.
            epochs: Number of training epochs.
            val_data: Optional validation data.
            batch_size: Training batch size.

        Returns:
            Self.
        """
        encoded_data, user_map, item_map = train_data.encode_ids()
        self._user_map = user_map
        self._item_map = item_map
        self._n_users = len(user_map)
        self._n_items = len(item_map)
        self._all_item_ids = train_data.item_ids

        arrays = encoded_data.to_numpy()
        user_ids = arrays["user_id"]
        item_ids = arrays["item_id"]

        # Build interaction lookup
        self._user_interactions = {}
        for u, i in zip(user_ids, item_ids, strict=False):
            self._user_interactions.setdefault(int(u), set()).add(int(i))

        # Build adjacency matrix
        self._adj_matrix = self._build_adj_matrix(user_ids, item_ids)

        # Initialize embeddings
        self._user_emb = nn.Embedding(self._n_users, self.embedding_dim).to(self.device)
        self._item_emb = nn.Embedding(self._n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self._user_emb.weight)
        nn.init.xavier_uniform_(self._item_emb.weight)

        optimizer = torch.optim.Adam(
            list(self._user_emb.parameters()) + list(self._item_emb.parameters()),
            lr=self.learning_rate,
        )

        n_interactions = len(user_ids)
        indices = np.arange(n_interactions)

        for _epoch in trange(epochs, desc="LightGCN Training"):
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            # Propagate once per epoch
            user_final, item_final = self._propagate()

            for start in range(0, n_interactions, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_users = user_ids[batch_idx]
                batch_pos_items = item_ids[batch_idx]

                # Negative sampling
                batch_neg_items = np.zeros_like(batch_pos_items)
                for i, u in enumerate(batch_users):
                    neg = np.random.randint(0, self._n_items)
                    user_items = self._user_interactions.get(int(u), set())
                    while neg in user_items:
                        neg = np.random.randint(0, self._n_items)
                    batch_neg_items[i] = neg

                u_idx = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                pi_idx = torch.tensor(batch_pos_items, dtype=torch.long, device=self.device)
                ni_idx = torch.tensor(batch_neg_items, dtype=torch.long, device=self.device)

                u_emb = user_final[u_idx]
                pi_emb = item_final[pi_idx]
                ni_emb = item_final[ni_idx]

                pos_scores = (u_emb * pi_emb).sum(dim=1)
                neg_scores = (u_emb * ni_emb).sum(dim=1)

                bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

                # L2 regularization on initial embeddings (not propagated)
                reg_loss = self.weight_decay * (
                    self._user_emb(u_idx).norm(2).pow(2)
                    + self._item_emb(pi_idx).norm(2).pow(2)
                    + self._item_emb(ni_idx).norm(2).pow(2)
                ) / len(batch_users)

                loss = bpr_loss + reg_loss

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
        if self._user_emb is None or self._item_emb is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        enc_users = np.array([self._user_map.get(int(u), -1) for u in user_ids])
        enc_items = np.array([self._item_map.get(int(i), -1) for i in item_ids])

        valid_mask = (enc_users >= 0) & (enc_items >= 0)
        scores = np.zeros(len(user_ids), dtype=np.float64)

        if valid_mask.any():
            user_final, item_final = self._propagate()

            u_idx = torch.tensor(enc_users[valid_mask], dtype=torch.long, device=self.device)
            i_idx = torch.tensor(enc_items[valid_mask], dtype=torch.long, device=self.device)

            u_emb = user_final[u_idx]
            i_emb = item_final[i_idx]
            scores[valid_mask] = (u_emb * i_emb).sum(dim=1).cpu().numpy()

        return scores
