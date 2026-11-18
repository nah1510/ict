"""PyTorch implementation of GRU4Rec with session-parallel minibatches."""

from typing import Iterator, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines import BaseSessionRecommender


def iter_session_parallel_batches(
    sessions: Sequence[Sequence[int]],
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (input_items, target_items, reset_mask) in session-parallel order.

    Each slot in the minibatch tracks one active session step-by-step. When a
    session finishes, reset_mask[slot] = True and the next session is loaded.
    """
    indices = np.arange(len(sessions))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    actual_bs = min(batch_size, len(sessions))
    iters = np.arange(actual_bs)
    offset_sessions = np.zeros(actual_bs, dtype=np.int64)
    next_session_ptr = actual_bs

    finished = False
    while not finished:
        x = np.zeros(actual_bs, dtype=np.int64)
        y = np.zeros(actual_bs, dtype=np.int64)
        reset_mask = np.zeros(actual_bs, dtype=bool)

        for slot in range(actual_bs):
            sess = sessions[indices[iters[slot]]]
            pos = offset_sessions[slot]
            if pos == 0:
                reset_mask[slot] = True
            x[slot] = sess[pos]
            y[slot] = sess[pos + 1]
            offset_sessions[slot] += 1

        yield x, y, reset_mask

        for slot in range(actual_bs):
            sess = sessions[indices[iters[slot]]]
            if offset_sessions[slot] >= len(sess) - 1:
                if next_session_ptr < len(sessions):
                    iters[slot] = next_session_ptr
                    offset_sessions[slot] = 0
                    next_session_ptr += 1
                else:
                    finished = True
                    break


def bpr_loss_in_batch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pairwise BPR loss using other targets in the minibatch as negative samples."""
    # logits: (B, n_items), targets: (B,)
    target_scores = logits[:, targets]  # (B, B): row i, col j = score of item targets[j] for sample i
    pos_scores = torch.diag(target_scores).unsqueeze(1)  # (B, 1)
    diff = pos_scores - target_scores
    return -F.logsigmoid(diff).mean()


def top1_loss_in_batch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """TOP1 ranking loss from Hidasi et al. (2016) using in-batch negative samples."""
    target_scores = logits[:, targets]  # (B, B)
    pos_scores = torch.diag(target_scores).unsqueeze(1)
    diff = target_scores - pos_scores
    loss = torch.sigmoid(diff) + torch.sigmoid(target_scores ** 2)
    return loss.mean()


class GRU4RecNet(nn.Module):
    """GRU4Rec neural architecture: Embedding -> Dropout -> GRU -> Linear."""

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 100,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, num_items + 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,) or (B, T)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        emb = self.dropout(self.item_emb(x))
        out, hidden_next = self.gru(emb, hidden)
        logits = self.out_proj(out[:, -1, :])
        return logits, hidden_next

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


class GRU4RecRecommender(BaseSessionRecommender):
    """Wrapper for training and inference with GRU4Rec."""

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 100,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 10,
        loss_type: str = "ce",
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_type = loss_type.lower()
        self.seed = seed
        self.name = f"GRU4Rec ({self.loss_type.upper()})"

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)
        self.model = GRU4RecNet(
            num_items=num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        self.loss_history: List[float] = []

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "ce":
            return F.cross_entropy(logits, targets, ignore_index=0)
        if self.loss_type == "bpr":
            return bpr_loss_in_batch(logits, targets)
        if self.loss_type == "top1":
            return top1_loss_in_batch(logits, targets)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def fit(self, train_sessions: Sequence[Sequence[int]]) -> "GRU4RecRecommender":
        valid_sessions = [s for s in train_sessions if len(s) >= 2]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_history = []

        self.model.train()
        for epoch in range(self.epochs):
            epoch_losses = []
            hidden = self.model.init_hidden(min(self.batch_size, len(valid_sessions)), self.device)

            for x_np, y_np, reset_np in iter_session_parallel_batches(
                valid_sessions,
                batch_size=self.batch_size,
                shuffle=True,
                seed=self.seed + epoch,
            ):
                if reset_np.any():
                    reset_idx = torch.as_tensor(reset_np, device=self.device)
                    hidden = hidden.clone()
                    hidden[:, reset_idx, :] = 0.0
                hidden = hidden.detach()

                x_t = torch.as_tensor(x_np, dtype=torch.long, device=self.device)
                y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits, hidden = self.model(x_t, hidden)
                loss = self._compute_loss(logits, y_t)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.loss_history.append(avg_loss)
        return self

    @torch.no_grad()
    def predict_scores(self, session_prefix: Sequence[int]) -> np.ndarray:
        self.model.eval()
        if not session_prefix:
            return np.zeros(self.num_items + 1, dtype=np.float32)
        x_t = torch.as_tensor(session_prefix, dtype=torch.long, device=self.device).unsqueeze(0)
        emb = self.model.item_emb(x_t)
        out, _ = self.model.gru(emb)
        logits = self.model.out_proj(out[:, -1, :]).squeeze(0)
        logits[0] = -1e9  # Mask padding token 0
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict_next(self, session_prefix: Sequence[int], top_k: int = 20) -> List[int]:
        scores = self.predict_scores(session_prefix)
        top_indices = np.argsort(-scores)[:top_k]
        return [int(idx) for idx in top_indices if idx > 0]

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "num_items": self.num_items,
                    "embed_dim": self.embed_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "loss_type": self.loss_type,
                },
                "loss_history": self.loss_history,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "GRU4RecRecommender":
        ckpt = torch.load(path, map_location=device or "cpu")
        cfg = ckpt["config"]
        rec = cls(
            num_items=cfg["num_items"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            loss_type=cfg["loss_type"],
            device=device,
        )
        rec.model.load_state_dict(ckpt["state_dict"])
        rec.loss_history = ckpt.get("loss_history", [])
        return rec
