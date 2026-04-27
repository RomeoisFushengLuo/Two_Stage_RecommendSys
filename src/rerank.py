# src/rerank_data.py
from __future__ import annotations

import os

# Must be set before importing numpy / torch / faiss
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm

from src.data import (
    load_events_from_reviews_gz,
    make_implicit_events_from_ratings,
    per_user_time_split,
    reindex_ids,
)
from src.two_tower_train import TwoTower


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    u = torch.tensor(users, dtype=torch.long, device=DEVICE)
    U = model.user_vec(u).detach().cpu().numpy().astype("float32")
    return l2_normalize(U)


def build_history_stats(hist_df: pd.DataFrame):
    """
    Build history-based lookup tables from a history dataframe.

    Returns:
      user_len[u]           = number of historical interactions
      item_cnt[i]           = item popularity in history
      ui_cnt[(u,i)]         = user-item frequency in history
      ui_last_ts[(u,i)]     = last timestamp user interacted with item in history
      user_last_item[u]     = user's last item in history
      user_last_ts[u]       = user's last timestamp in history
      cooc[(i_prev,i_next)] = adjacent-pair co-occurrence count
    """
    hist_df = hist_df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    user_len = hist_df.groupby("user_idx").size().to_dict()
    item_cnt = hist_df.groupby("item_idx").size().to_dict()

    ui_agg = (
        hist_df.groupby(["user_idx", "item_idx"])["timestamp"]
        .agg(["size", "max"])
        .reset_index()
    )

    ui_cnt = {
        (int(r.user_idx), int(r.item_idx)): int(r.size)
        for r in ui_agg.itertuples(index=False)
    }

    ui_last_ts = {
        (int(r.user_idx), int(r.item_idx)): int(r.max)
        for r in ui_agg.itertuples(index=False)
    }

    last_rows = hist_df.groupby("user_idx").tail(1)
    user_last_item = dict(
        zip(last_rows["user_idx"].astype(int), last_rows["item_idx"].astype(int))
    )
    user_last_ts = dict(
        zip(last_rows["user_idx"].astype(int), last_rows["timestamp"].astype(int))
    )

    cooc = defaultdict(int)
    for _, g in hist_df.groupby("user_idx", sort=False):
        items = g["item_idx"].to_numpy()
        if len(items) < 2:
            continue

        for a, b in zip(items[:-1], items[1:]):
            a = int(a)
            b = int(b)
            cooc[(a, b)] += 1
            cooc[(b, a)] += 1

    return user_len, item_cnt, ui_cnt, ui_last_ts, user_last_item, user_last_ts, cooc


def maybe_append_positive(
    cands: np.ndarray,
    sims: np.ndarray,
    pos_item: int,
    append_positive: bool,
):
    """
    If append_positive=True and pos_item is missing, append it as an artificial row.

    Use case:
      - Training reranker: append_positive=True can help create valid ranking groups.
      - Testing reranker: append_positive=False to avoid leakage.
    """
    if pos_item in cands:
        return cands, sims, True

    if not append_positive:
        return cands, sims, False

    cands2 = np.append(cands, pos_item).astype(np.int64)
    sims2 = np.append(sims, -1e9).astype(np.float32)
    return cands2, sims2, False


def write_rerank_csv(
    out_csv: str,
    model: TwoTower,
    index: faiss.Index,
    base_hist_df: pd.DataFrame,
    target_df: pd.DataFrame,
    K: int = 500,
    batch_users: int = 1024,
    append_positive: bool = False,
):
    """
    Build reranker dataset.

    base_hist_df:
      History used to compute features. Must be strictly before target period.

    target_df:
      One target item per user, e.g. val or test.

    append_positive:
      True for reranker training if you want every group to contain a positive.
      False for reranker testing to avoid leakage.
    """
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_csv):
        os.remove(out_csv)

    (
        user_len,
        item_cnt,
        ui_cnt,
        ui_last_ts,
        user_last_item,
        user_last_ts,
        cooc,
    ) = build_history_stats(base_hist_df)

    tgt_user = target_df["user_idx"].to_numpy(dtype=np.int64)
    tgt_item = target_df["item_idx"].to_numpy(dtype=np.int64)
    tgt_ts = target_df["timestamp"].to_numpy(dtype=np.int64)

    gt = {
        int(u): (int(i), int(t))
        for u, i, t in zip(tgt_user, tgt_item, tgt_ts)
    }

    users = tgt_user.copy()
    header_written = False

    retrieval_hits = 0
    total_users = 0
    total_rows = 0
    positive_rows = 0

    for start in tqdm(range(0, len(users), batch_users), desc=f"Build rerank rows K={K}"):
        end = min(start + batch_users, len(users))
        batch_u = users[start:end]

        U = build_user_matrix(model, batch_u)
        sims, I = index.search(U, K)

        rows = []

        for bi, u in enumerate(batch_u):
            u = int(u)
            pos_item, pos_ts = gt[u]

            cands = I[bi].astype(np.int64)
            simv = sims[bi].astype(np.float32)

            retrieved_hit = pos_item in cands
            if retrieved_hit:
                retrieval_hits += 1
            total_users += 1

            cands2, simv2, _ = maybe_append_positive(
                cands=cands,
                sims=simv,
                pos_item=pos_item,
                append_positive=append_positive,
            )

            last_it = user_last_item.get(u, -1)
            hist_end_ts = user_last_ts.get(u, 0)
            ulen = int(user_len.get(u, 0))

            for j, it in enumerate(cands2):
                it = int(it)
                label = 1 if it == pos_item else 0

                if label == 1:
                    positive_rows += 1

                sim = float(simv2[j])

                user_freq = ulen
                item_freq = int(item_cnt.get(it, 0))
                ui_freq = int(ui_cnt.get((u, it), 0))

                last_ui_ts = ui_last_ts.get((u, it), None)

                # Use history end timestamp rather than target timestamp
                # to reduce leakage from target time.
                if last_ui_ts is None or hist_end_ts == 0:
                    ui_recency_days = 1e9
                else:
                    ui_recency_days = float((hist_end_ts - int(last_ui_ts)) / 86400.0)

                if last_it == -1:
                    cooc_with_last = 0
                else:
                    cooc_with_last = int(cooc.get((last_it, it), 0))

                # If appended positive exists, its j == K and retrieved = 0.
                retrieved_flag = 1 if j < K else 0
                retrieval_rank = j if j < K else K

                if hist_end_ts:
                    gap_from_last_event_days = float((pos_ts - hist_end_ts) / 86400.0)
                else:
                    gap_from_last_event_days = 1e9

                rows.append(
                    (
                        u,
                        it,
                        label,
                        sim,
                        retrieved_flag,
                        retrieval_rank,
                        user_freq,
                        item_freq,
                        ui_freq,
                        ui_recency_days,
                        cooc_with_last,
                        gap_from_last_event_days,
                    )
                )

        df_out = pd.DataFrame(
            rows,
            columns=[
                "user_idx",
                "item_idx",
                "label",
                "sim",
                "retrieved",
                "retrieval_rank",
                "user_freq",
                "item_freq",
                "ui_freq",
                "ui_recency_days",
                "cooc_with_last",
                "gap_from_last_event_days",
            ],
        )

        df_out.to_csv(out_csv, mode="a", header=not header_written, index=False)
        header_written = True

        total_rows += len(df_out)

    print(f"Saved -> {out_csv}")
    print(f"Rows: {total_rows:,}")
    print(f"Positive rows: {positive_rows:,}")
    print(f"Retrieval hit rate@{K}: {retrieval_hits / max(total_users, 1):.4f}")
    print(f"append_positive={append_positive}")


def resolve_project_paths():
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(SRC_DIR)

    gz_path = os.path.join(SRC_DIR, "data", "Electronics_5.json.gz")

    ckpt_candidates = [
        os.path.join(ROOT, "outputs", "two_tower.pt"),
        os.path.join(SRC_DIR, "outputs", "two_tower.pt"),
    ]

    ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)

    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Missing data file: {gz_path}")

    if ckpt_path is None:
        raise FileNotFoundError(
            "Missing checkpoint. Tried:\n" + "\n".join(ckpt_candidates)
        )

    return gz_path, ckpt_path, out_dir


def main():
    gz_path, ckpt_path, out_dir = resolve_project_paths()

    # 1) Load data
    df_ratings = load_events_from_reviews_gz(gz_path, max_rows=2_000_000)

    events = make_implicit_events_from_ratings(df_ratings, positive_rule="rating>=4")
    train, val, test = per_user_time_split(events, min_interactions=5)
    train, val, test, user2idx, item2idx = reindex_ids(train, val, test)

    print(
        f"Users={len(user2idx)} Items={len(item2idx)} "
        f"Train={len(train)} Val={len(val)} Test={len(test)}"
    )

    # 2) Load two-tower model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TwoTower(ckpt["n_users"], ckpt["n_items"], dim=ckpt["dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3) Build FAISS index
    item_mat = build_item_matrix(model)
    d = item_mat.shape[1]

    index = faiss.IndexFlatIP(d)
    index.add(item_mat)

    # 4) Build reranker datasets
    K = 500

    train_out = os.path.join(out_dir, "rerank_train_k500.csv")
    test_out = os.path.join(out_dir, "rerank_test_k500.csv")

    # Train reranker:
    # append_positive=True makes training groups usable even if retrieval missed the val item.
    write_rerank_csv(
        out_csv=train_out,
        model=model,
        index=index,
        base_hist_df=train,
        target_df=val,
        K=K,
        batch_users=1024,
        append_positive=True,
    )

    # Test reranker:
    # append_positive=False avoids leakage. If retrieval missed, reranker cannot recover.
    hist_tv = pd.concat([train, val], ignore_index=True)
    write_rerank_csv(
        out_csv=test_out,
        model=model,
        index=index,
        base_hist_df=hist_tv,
        target_df=test,
        K=K,
        batch_users=1024,
        append_positive=False,
    )


if __name__ == "__main__":
    main()