# src/faiss_eval.py
import os

# MUST set these BEFORE importing numpy/torch/faiss
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import torch

from src.data import (
    load_events_from_reviews_gz,
    make_implicit_events_from_ratings,
    per_user_time_split,
    reindex_ids,
)
from src.two_tower_train import TwoTower


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


@torch.no_grad()
def build_item_matrix(model: TwoTower) -> np.ndarray:
    model.eval()
    W = model.item_matrix().detach().cpu().numpy().astype("float32")
    return l2_normalize(W)


@torch.no_grad()
def build_user_matrix(model: TwoTower, users: np.ndarray) -> np.ndarray:
    model.eval()
    u = torch.tensor(users, dtype=torch.long)
    U = model.user_vec(u).detach().cpu().numpy().astype("float32")
    return l2_normalize(U)


def recall_at_k_faiss(index, user_mat: np.ndarray, gt_items: np.ndarray, k: int) -> float:
    _, I = index.search(user_mat, k)
    hit = 0
    for row, gt in zip(I, gt_items):
        if int(gt) in row:
            hit += 1
    return hit / max(len(gt_items), 1)


def main():
    import faiss  # delay import a bit

    SRC_DIR = os.path.dirname(os.path.abspath(__file__))     # .../src
    ROOT = os.path.dirname(SRC_DIR)                          # project root

    gz_path = os.path.join(SRC_DIR, "data", "Electronics_5.json.gz")
    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Missing file: {gz_path}")

    ckpt_path = os.path.join(SRC_DIR, "outputs", "two_tower.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path} (run two_tower_train first)")

    # 1) load & split data (same preprocessing as training)
    df_ratings = load_events_from_reviews_gz(gz_path, max_rows=2_000_000)
    events = make_implicit_events_from_ratings(df_ratings, positive_rule="rating>=4")
    train, val, test = per_user_time_split(events, min_interactions=5)
    train, val, test, user2idx, item2idx = reindex_ids(train, val, test)

    # 2) load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TwoTower(ckpt["n_users"], ckpt["n_items"], dim=ckpt["dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3) eval users
    test_users = test["user_idx"].to_numpy(dtype=np.int64)
    test_items = test["item_idx"].to_numpy(dtype=np.int64)

    # 4) matrices
    item_mat = build_item_matrix(model)
    user_mat = build_user_matrix(model, test_users)

    d = item_mat.shape[1]
    print(f"Eval users={len(test_users)} items={item_mat.shape[0]} dim={d}")

    # 5) FlatIP
    index_flat = faiss.IndexFlatIP(d)
    index_flat.add(item_mat)
    t0 = time.time()
    r50 = recall_at_k_faiss(index_flat, user_mat, test_items, k=50)
    t1 = time.time()
    print(f"[FAISS FlatIP] Recall@50={r50:.4f}  time={t1-t0:.3f}s")

    # 6) HNSW
    index_hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efSearch = 64
    index_hnsw.add(item_mat)
    t0 = time.time()
    r50_h = recall_at_k_faiss(index_hnsw, user_mat, test_items, k=50)
    t1 = time.time()
    print(f"[FAISS HNSW]  Recall@50={r50_h:.4f}  time={t1-t0:.3f}s  (efSearch=64)")


if __name__ == "__main__":
    main()