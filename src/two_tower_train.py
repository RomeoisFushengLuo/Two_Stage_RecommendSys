from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.data import (
    load_events_from_reviews_gz,
    make_implicit_events_from_ratings,
    per_user_time_split,
    reindex_ids,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PairDataset(Dataset):
    """Each sample is (user_idx, item_idx) positive interaction."""
    def __init__(self, df: pd.DataFrame):
        self.u = df["user_idx"].to_numpy(dtype=np.int64)
        self.i = df["item_idx"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.u)

    def __getitem__(self, idx: int):
        return self.u[idx], self.i[idx]


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor):
        u = self.user_emb(u_idx)  # [B, D]
        v = self.item_emb(i_idx)  # [B, D]
        return u, v

    @torch.no_grad()
    def user_vec(self, u_idx: torch.Tensor):
        return self.user_emb(u_idx)

    @torch.no_grad()
    def item_matrix(self):
        return self.item_emb.weight  # [I, D]


def info_nce_inbatch_loss(u: torch.Tensor, v: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    In-batch negatives (InfoNCE / sampled softmax):
    logits = (u_norm @ v_norm^T) / tau, target = diag indices.
    """
    u = nn.functional.normalize(u, dim=1)
    v = nn.functional.normalize(v, dim=1)
    logits = (u @ v.T) / temperature
    targets = torch.arange(u.size(0), device=u.device)
    return nn.functional.cross_entropy(logits, targets)


@torch.no_grad()
def recall_at_k(model: TwoTower, test_df: pd.DataFrame, k: int = 50, max_users: int = 2000, batch_users: int = 256) -> float:
    """
    Offline recall@K:
      For each user in test, score all items by dot(user_emb, item_emb),
      and check if true test item is in topK.
    Batched over users for speed.
    """
    model.eval()

    users = test_df["user_idx"].unique()
    if len(users) > max_users:
        users = np.random.default_rng(0).choice(users, size=max_users, replace=False)

    # ground truth: last-1 split => 1 test item per user
    gt = dict(zip(test_df["user_idx"].to_list(), test_df["item_idx"].to_list()))

    item_mat = model.item_matrix().to(DEVICE)  # [I, D]

    hit = 0
    total = 0

    for s in range(0, len(users), batch_users):
        batch = users[s : s + batch_users]
        u_t = torch.tensor(batch, device=DEVICE, dtype=torch.long)  # [B]
        u_vec = model.user_vec(u_t)  # [B, D]
        scores = u_vec @ item_mat.T  # [B, I]

        topk = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()  # [B, K]

        for u, row in zip(batch, topk):
            if gt[int(u)] in row:
                hit += 1
            total += 1

    return hit / max(total, 1)


def main():
    # assumes project root contains "src/data/Electronics_5.json.gz"

    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    gz_path = os.path.join(SRC_DIR, "data", "Electronics_5.json.gz")
    assert os.path.exists(gz_path), f"Missing file: {gz_path}"
    
    df_ratings = load_events_from_reviews_gz(gz_path, max_rows=2_000_000)

    events = make_implicit_events_from_ratings(df_ratings, positive_rule="rating>=4")
    train, val, test = per_user_time_split(events, min_interactions=5)
    train, val, test, user2idx, item2idx = reindex_ids(train, val, test)

    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"Users={n_users} Items={n_items} Train={len(train)} Val={len(val)} Test={len(test)}")

    # ---- model + loader ----
    dim = 64
    batch_size = 2048
    epochs = 3
    lr = 2e-3

    ds = PairDataset(train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = TwoTower(n_users, n_items, dim=dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # ---- training ----
    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for u_idx, i_idx in dl:
            u_idx = u_idx.to(DEVICE)
            i_idx = i_idx.to(DEVICE)

            u, v = model(u_idx, i_idx)
            loss = info_nce_inbatch_loss(u, v, temperature=0.07)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        r50 = recall_at_k(model, test, k=50, max_users=2000, batch_users=256)
        print(f"Epoch {ep}: loss={avg_loss:.4f}  Recall@50≈{r50:.4f} (on 2k users)")

    # ---- save ----
    out_dir = os.path.join(SRC_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "two_tower.pt")

    torch.save(
        {"state_dict": model.state_dict(), "n_users": n_users, "n_items": n_items, "dim": dim},
        out_path,
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()