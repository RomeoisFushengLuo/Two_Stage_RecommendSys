from __future__ import annotations
import gzip
import json
import os
import ssl
import certifi
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np 
import pandas as pd 
from tqdm import tqdm 

# Amazon Reviews 2018 (a.k.a. nijianmo / amazon_v2), with "Electronic" category
REVIEWS_URL = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz"

def download_file(url: str, out_dir: str = "data", chunk_size: int = 1 << 20) -> str:
    """
    Robust downloader for https urls on macOS/conda:
    - Uses certifi CA bundle to avoid SSL issues
    - Streams download to disk (no huge memory)
    - Works even when urllib.urlretrieve doesn't support context=
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = url.split("/")[-1]
    path = os.path.join(out_dir, fname)

    # If already downloaded, reuse
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    print(f"Downloading -> {path}")

    ctx = ssl.create_default_context(cafile=certifi.where())
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(req, context=ctx, timeout=60) as r, open(path, "wb") as f:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
    except (HTTPError, URLError) as e:
        # Clean up partial file
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise RuntimeError(f"Download failed: {url}\nError: {e}") from e

    return path

def load_events_from_reviews_gz(
    path: str,
    max_rows: int = 2_000_000,   
) -> pd.DataFrame:
    """
    Stream-read Amazon reviews .json.gz (one JSON per line),
    and keep only essential columns for implicit recommendation:
    user_id, item_id, timestamp, rating.

    This avoids loading the full raw JSON into memory.
    """
    users, items, ts, rating = [], [], [], []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Reading reviews")):
            obj = json.loads(line)

            # defensive: skip incomplete rows
            u = obj.get("reviewerID")
            it = obj.get("asin")
            t = obj.get("unixReviewTime")
            r = obj.get("overall")

            if u is None or it is None or t is None or r is None:
                continue

            users.append(u)
            items.append(it)
            ts.append(int(t))
            rating.append(float(r))

            if max_rows is not None and len(users) >= max_rows:
                break

    df = pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "timestamp": ts,
        "rating": rating
    })
    return df


def make_implicit_events_from_ratings(
    df: pd.DataFrame,
    positive_rule: str = "rating>=4",
) -> pd.DataFrame:
    """
    Convert to implicit events with optional weighting.
    Output: user_id, item_id, timestamp, label, weight
    """
    out = df.copy()

    if positive_rule == "rating>=1":
        out["label"] = 1
        out["weight"] = 1.0
        return out[["user_id", "item_id", "timestamp", "label", "weight"]]

    if positive_rule == "rating>=4":
        out = out[out["rating"] >= 4].copy()
        out["label"] = 1
        out["weight"] = 1.0
        return out[["user_id", "item_id", "timestamp", "label", "weight"]]

    if positive_rule == "weight=rating":
        out["label"] = 1
        out["weight"] = out["rating"]
        return out[["user_id", "item_id", "timestamp", "label", "weight"]]

    raise ValueError(f"Unknown positive_rule: {positive_rule}")


def per_user_time_split(
    events: pd.DataFrame,
    min_interactions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict per-user split:
      - Train: all but last 2
      - Val: second last
      - Test: last
    """
    events = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    cnt = events.groupby("user_id").size()
    keep_users = cnt[cnt >= min_interactions].index
    events = events[events["user_id"].isin(keep_users)].copy()

    events["rank"] = events.groupby("user_id").cumcount()
    events["cnt"] = events.groupby("user_id")["rank"].transform("max") + 1

    test_mask = events["rank"] == (events["cnt"] - 1)
    val_mask = events["rank"] == (events["cnt"] - 2)
    train_mask = ~(test_mask | val_mask)

    train = events[train_mask].drop(columns=["rank", "cnt"]).reset_index(drop=True)
    val = events[val_mask].drop(columns=["rank", "cnt"]).reset_index(drop=True)
    test = events[test_mask].drop(columns=["rank", "cnt"]).reset_index(drop=True)
    return train, val, test


def reindex_ids(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Map string IDs to contiguous indices for embeddings.
    Fit mappings on TRAIN ONLY.
    """
    user2idx = {u: i for i, u in enumerate(train["user_id"].unique())}
    item2idx = {it: i for i, it in enumerate(train["item_id"].unique())}

    def _map(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["user_idx"] = out["user_id"].map(user2idx)
        out["item_idx"] = out["item_id"].map(item2idx)
        out = out.dropna(subset=["user_idx", "item_idx"])
        out["user_idx"] = out["user_idx"].astype(int)
        out["item_idx"] = out["item_idx"].astype(int)
        return out

    return _map(train), _map(val), _map(test), user2idx, item2idx


if __name__ == "__main__":
    gz_path = download_file(REVIEWS_URL, out_dir="data")

    # First Trail == Fewer Samples
    df_ratings = load_events_from_reviews_gz(gz_path, max_rows=2_000_000)

    events = make_implicit_events_from_ratings(df_ratings, positive_rule="rating>=4")

    train, val, test = per_user_time_split(events, min_interactions=5)
    train, val, test, user2idx, item2idx = reindex_ids(train, val, test)

    print("Users:", len(user2idx), "Items:", len(item2idx))
    print("Train:", len(train), "Val:", len(val), "Test:", len(test))
    print(train.head())