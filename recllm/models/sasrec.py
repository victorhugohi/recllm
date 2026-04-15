"""Self-Attentive Sequential Recommendation (SASRec).

Reference: Kang & McAuley (2018). Self-Attentive Sequential Recommendation. ICDM 2018.
Uses causal self-attention over user interaction sequences.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from recllm.data.base import InteractionData
from recllm.models.base import BaseModel


class _SASRecNetwork(nn.Module):
    """SASRec transformer network."""

    def __init__(
        self,
        n_items: int,
        embedding_dim: int,
        max_seq_len: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len

        # Item and positional embeddings
        # item 0 is reserved as padding
        self.item_emb = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, item_seq: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through SASRec.

        Args:
            item_seq: (batch, seq_len) item ID sequences (0-padded).
            positions: (batch, seq_len) position indices.

        Returns:
            (batch, seq_len, embedding_dim) sequence representations.
        """
        seq_emb = self.item_emb(item_seq) + self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        # Causal mask: each position can only attend to itself and earlier
        seq_len = item_seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device), diagonal=1
        ).bool()

        # Padding mask: True where item_seq == 0
        padding_mask = item_seq == 0

        output = self.transformer(
            seq_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.layer_norm(output)

    def predict_next(
        self, item_seq: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Predict scores for all items given a sequence.

        Args:
            item_seq: (batch, seq_len) sequences.
            positions: (batch, seq_len) positions.

        Returns:
            (batch, n_items+1) scores for each item.
        """
        seq_output = self.forward(item_seq, positions)  # (B, L, D)
        last_hidden = seq_output[:, -1, :]  # (B, D) -- last position
        # Score all items by dot product
        all_item_emb = self.item_emb.weight  # (n_items+1, D)
        scores = last_hidden @ all_item_emb.T  # (B, n_items+1)
        return scores


class SASRec(BaseModel):
    """Self-Attentive Sequential Recommendation.

    Models user interaction sequences with causal self-attention.
    Predicts the next item a user will interact with given their
    history. Requires a 'timestamp' column for ordering.

    Args:
        embedding_dim: Item and position embedding dimension.
        max_seq_len: Maximum sequence length to consider.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
        learning_rate: Optimizer learning rate.
        device: PyTorch device.

    Example:
        >>> model = SASRec(embedding_dim=64, max_seq_len=50)
        >>> model.fit(train_data, epochs=100)
        >>> recs = model.recommend(user_id=1, n=10)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_seq_len: int = 50,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        device: str = "auto",
    ):
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model: _SASRecNetwork | None = None
        self._all_item_ids: np.ndarray = np.array([])
        self._user_sequences: dict[int, list[int]] = {}
        self._user_interactions: dict[int, set[int]] = {}

    def _build_sequences(self, data: InteractionData) -> dict[int, list[int]]:
        """Build ordered interaction sequences per user."""
        df = data.interactions
        if "timestamp" in df.columns:
            df = df.sort(["user_id", "timestamp"])
        else:
            df = df.sort("user_id")

        arrays = df.to_dict()
        user_ids = arrays["user_id"].to_list()
        item_ids = arrays["item_id"].to_list()

        sequences: dict[int, list[int]] = {}
        for u, i in zip(user_ids, item_ids):
            sequences.setdefault(u, []).append(i)
        return sequences

    def _pad_sequence(self, seq: list[int]) -> tuple[list[int], list[int]]:
        """Pad/truncate sequence to max_seq_len. Returns (items, positions)."""
        seq = seq[-self.max_seq_len :]
        length = len(seq)
        padding = [0] * (self.max_seq_len - length)
        items = padding + seq
        positions = list(range(self.max_seq_len))
        return items, positions

    def fit(
        self,
        train_data: InteractionData,
        epochs: int = 100,
        val_data: InteractionData | None = None,
        batch_size: int = 128,
    ) -> Self:
        """Train SASRec on sequential interaction data.

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
        self._reverse_item_map = {v: k for k, v in item_map.items()}
        n_items = len(item_map)
        self._all_item_ids = train_data.item_ids

        # Shift item IDs by +1 (0 = padding)
        self._item_shift = 1

        # Build sequences with encoded IDs
        sequences = self._build_sequences(encoded_data)
        # Shift IDs: encoded 0..n-1 -> 1..n
        for uid in sequences:
            sequences[uid] = [iid + self._item_shift for iid in sequences[uid]]

        self._user_sequences = sequences

        # Also store interaction sets for recommend() exclusion
        self._user_interactions = {}
        for uid, seq in sequences.items():
            self._user_interactions[uid] = set(seq)

        self._model = _SASRecNetwork(
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.max_seq_len,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )

        user_ids_list = list(sequences.keys())

        for epoch in trange(epochs, desc="SASRec Training"):
            self._model.train()
            np.random.shuffle(user_ids_list)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(user_ids_list), batch_size):
                batch_users = user_ids_list[start : start + batch_size]

                batch_seqs = []
                batch_pos_items = []
                batch_neg_items = []

                for uid in batch_users:
                    seq = sequences[uid]
                    if len(seq) < 2:
                        continue
                    # Input: all but last; target: last item
                    input_seq = seq[:-1]
                    target = seq[-1]

                    items, positions = self._pad_sequence(input_seq)
                    batch_seqs.append((items, positions))
                    batch_pos_items.append(target)

                    # Negative sample
                    neg = np.random.randint(1, n_items + 1)
                    user_items = self._user_interactions.get(uid, set())
                    while neg in user_items:
                        neg = np.random.randint(1, n_items + 1)
                    batch_neg_items.append(neg)

                if not batch_seqs:
                    continue

                item_seqs = torch.tensor(
                    [s[0] for s in batch_seqs], dtype=torch.long, device=self.device
                )
                pos_seqs = torch.tensor(
                    [s[1] for s in batch_seqs], dtype=torch.long, device=self.device
                )
                pos_items = torch.tensor(
                    batch_pos_items, dtype=torch.long, device=self.device
                )
                neg_items = torch.tensor(
                    batch_neg_items, dtype=torch.long, device=self.device
                )

                # Forward
                seq_output = self._model.forward(item_seqs, pos_seqs)
                last_hidden = seq_output[:, -1, :]  # (B, D)

                pos_emb = self._model.item_emb(pos_items)
                neg_emb = self._model.item_emb(neg_items)

                pos_scores = (last_hidden * pos_emb).sum(dim=1)
                neg_scores = (last_hidden * neg_emb).sum(dim=1)

                loss = -torch.log(
                    torch.sigmoid(pos_scores - neg_scores) + 1e-8
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        self._model.eval()
        return self

    @torch.no_grad()
    def predict(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        """Predict scores for user-item pairs using sequence representations."""
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        scores = np.zeros(len(user_ids), dtype=np.float64)

        # Group by user for efficiency
        user_to_indices: dict[int, list[int]] = {}
        for idx, u in enumerate(user_ids):
            enc_u = self._user_map.get(int(u), -1)
            if enc_u >= 0:
                user_to_indices.setdefault(enc_u, []).append(idx)

        for enc_uid, indices in user_to_indices.items():
            seq = self._user_sequences.get(enc_uid, [])
            if not seq:
                continue

            items, positions = self._pad_sequence(seq)
            item_seq = torch.tensor([items], dtype=torch.long, device=self.device)
            pos_seq = torch.tensor([positions], dtype=torch.long, device=self.device)

            seq_output = self._model.forward(item_seq, pos_seq)
            last_hidden = seq_output[0, -1, :]  # (D,)

            for idx in indices:
                enc_item = self._item_map.get(int(item_ids[idx]), -1)
                if enc_item >= 0:
                    shifted_item = enc_item + self._item_shift
                    item_emb = self._model.item_emb.weight[shifted_item]
                    scores[idx] = (last_hidden * item_emb).sum().cpu().item()

        return scores
