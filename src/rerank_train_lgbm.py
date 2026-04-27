from __future__ import annotations

import os
import numpy as np
import pandas as pd
import lightgbm as lgb


FEATURES = [
    "sim",
    "retrieval_rank",
    "user_freq",
    "item_freq",
    "ui_freq",
    "ui_recency_days",
    "cooc_with_last",
    "gap_from_last_event_days",
]


def resolve_paths():
    """
    Resolve project paths robustly.

    Expected:
      project_root/
        outputs/
          rerank_train_k500.csv
          rerank_test_k500.csv
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src_dir)

    outputs_dir = os.path.join(root, "outputs")
    train_path = os.path.join(outputs_dir, "rerank_train_k500.csv")
    test_path = os.path.join(outputs_dir, "rerank_test_k500.csv")
    model_path = os.path.join(outputs_dir, "reranker_lgbm.txt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train rerank file: {train_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test rerank file: {test_path}")

    return train_path, test_path, model_path


def sort_for_lgb_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    LightGBM ranking requires rows from the same query/user to be contiguous.
    """
    return (
        df.sort_values(["user_idx", "retrieval_rank"], kind="stable")
        .reset_index(drop=True)
    )


def filter_groups_with_positive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only user groups where at least one candidate has label=1.

    This is necessary for LambdaRank training/conditional NDCG evaluation.
    Groups with all label=0 provide no ranking signal.
    """
    return (
        df.groupby("user_idx", group_keys=False)
        .filter(lambda g: g["label"].sum() >= 1)
        .reset_index(drop=True)
    )


def make_groups(df: pd.DataFrame) -> np.ndarray:
    """
    LightGBM group array: number of candidate rows per user/query.
    """
    return df.groupby("user_idx", sort=False).size().to_numpy(dtype=np.int32)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast features to float32 and handle extreme values.
    """
    out = df.copy()

    for c in FEATURES:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = out[c].fillna(1e9)
        out[c] = out[c].astype(np.float32)

    out["label"] = out["label"].astype(np.int32)
    return out


def ndcg_at_k_from_scores(df: pd.DataFrame, scores: np.ndarray, k: int) -> float:
    """
    Mean NDCG@K.

    With one positive target item per user:
      - if positive appears at rank r, DCG = 1 / log2(r + 1)
      - ideal DCG = 1
      - if no positive appears in top-K, NDCG = 0
    """
    df = df.copy()
    df["score"] = scores

    ndcgs = []

    for _, g in df.groupby("user_idx", sort=False):
        g = g.sort_values("score", ascending=False).head(k)

        pos_rank = np.where(g["label"].to_numpy() == 1)[0]

        if len(pos_rank) == 0:
            ndcgs.append(0.0)
        else:
            rank = int(pos_rank[0]) + 1
            ndcgs.append(1.0 / np.log2(rank + 1.0))

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def recall_at_k_from_scores(df: pd.DataFrame, scores: np.ndarray, k: int) -> float:
    """
    Recall@K / HitRate@K after reranking.

    For each user, check whether the true item is in top-K after sorting by score.
    """
    df = df.copy()
    df["score"] = scores

    hit = 0
    total = 0

    for _, g in df.groupby("user_idx", sort=False):
        topk = g.sort_values("score", ascending=False).head(k)

        if topk["label"].max() == 1:
            hit += 1

        total += 1

    return hit / max(total, 1)


def retrieval_hit_rate(df: pd.DataFrame) -> float:
    """
    Retrieval hit rate over the candidate set.

    This answers:
      Did the true target item appear anywhere in the retrieved candidate set?
    """
    hit = (
        df.groupby("user_idx")["label"]
        .max()
        .sum()
    )

    total = df["user_idx"].nunique()
    return float(hit / max(total, 1))


def evaluate_block(name: str, df: pd.DataFrame, scores: np.ndarray):
    """
    Print NDCG/Recall metrics for a given dataframe.
    """
    ndcg10 = ndcg_at_k_from_scores(df, scores, k=10)
    ndcg50 = ndcg_at_k_from_scores(df, scores, k=50)
    r50 = recall_at_k_from_scores(df, scores, k=50)

    print(f"{name:<28} NDCG@10={ndcg10:.4f}  NDCG@50={ndcg50:.4f}  Recall@50={r50:.4f}")


def main():
    train_path, test_path, model_path = resolve_paths()

    print("Loading reranker data...")
    df_tr_raw = pd.read_csv(train_path)
    df_te_raw = pd.read_csv(test_path)

    print(f"Raw train rows: {len(df_tr_raw):,}, users: {df_tr_raw['user_idx'].nunique():,}")
    print(f"Raw test rows:  {len(df_te_raw):,}, users: {df_te_raw['user_idx'].nunique():,}")

    # ------------------------------------------------------------------
    # Important leakage control
    # ------------------------------------------------------------------
    # If rerank_train_k500.csv was generated with append_positive=True,
    # then artificial positives have retrieved=0 and retrieval_rank=K.
    #
    # For a clean reranker, train only on naturally retrieved candidates:
    #   retrieved == 1
    #
    # Then keep only groups where the positive item was actually retrieved.
    # ------------------------------------------------------------------
    if "retrieved" in df_tr_raw.columns:
        df_tr = df_tr_raw[df_tr_raw["retrieved"] == 1].copy()
    else:
        df_tr = df_tr_raw.copy()

    df_tr = filter_groups_with_positive(df_tr)
    df_tr = sort_for_lgb_group(df_tr)
    df_tr = prepare_features(df_tr)

    # Test set should already have append_positive=False.
    # We keep the full test set for unconditional end-to-end metrics.
    df_te_all = sort_for_lgb_group(df_te_raw.copy())
    df_te_all = prepare_features(df_te_all)

    # Conditional test set: only users where FAISS actually retrieved the true item.
    # This evaluates ranking quality given candidate recall succeeded.
    df_te_cond = filter_groups_with_positive(df_te_all)
    df_te_cond = sort_for_lgb_group(df_te_cond)
    df_te_cond = prepare_features(df_te_cond)

    print("\nAfter leakage-safe filtering:")
    print(f"Train rows: {len(df_tr):,}, users with positive: {df_tr['user_idx'].nunique():,}")
    print(f"Test rows:  {len(df_te_all):,}, users total: {df_te_all['user_idx'].nunique():,}")
    print(f"Test conditional rows: {len(df_te_cond):,}, users with positive: {df_te_cond['user_idx'].nunique():,}")

    print(f"\nRetrieval HitRate@500 on test: {retrieval_hit_rate(df_te_all):.4f}")

    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float32)
    y_tr = df_tr["label"].to_numpy(dtype=np.int32)
    g_tr = make_groups(df_tr)

    X_val = df_te_cond[FEATURES].to_numpy(dtype=np.float32)
    y_val = df_te_cond["label"].to_numpy(dtype=np.int32)
    g_val = make_groups(df_te_cond)

    train_set = lgb.Dataset(
        X_tr,
        label=y_tr,
        group=g_tr,
        feature_name=FEATURES,
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X_val,
        label=y_val,
        group=g_val,
        feature_name=FEATURES,
        reference=train_set,
        free_raw_data=False,
    )

    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10, 50],
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
    )

    print("\nTraining LightGBM LambdaRank...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        valid_names=["test_conditional"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(50),
        ],
    )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    print("\nScoring...")

    # Full test: unconditional end-to-end metrics
    X_te_all = df_te_all[FEATURES].to_numpy(dtype=np.float32)
    scores_all = model.predict(X_te_all, num_iteration=model.best_iteration)

    # Conditional test: ranking quality given retrieval hit
    X_te_cond = df_te_cond[FEATURES].to_numpy(dtype=np.float32)
    scores_cond = model.predict(X_te_cond, num_iteration=model.best_iteration)

    # Baseline: two-tower similarity only
    base_scores_all = df_te_all["sim"].to_numpy(dtype=np.float32)
    base_scores_cond = df_te_cond["sim"].to_numpy(dtype=np.float32)

    print("\n=== Test Results ===")
    print("Unconditional metrics include users where retrieval missed the true item.")
    evaluate_block("LambdaRank full test", df_te_all, scores_all)
    evaluate_block("TwoTower sim full test", df_te_all, base_scores_all)

    print("\nConditional metrics only include users where true item is in Top-500 candidates.")
    evaluate_block("LambdaRank conditional", df_te_cond, scores_cond)
    evaluate_block("TwoTower sim conditional", df_te_cond, base_scores_cond)

    # Feature importance
    print("\n=== Feature Importance ===")
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    print(importance.to_string(index=False))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"\nSaved -> {model_path}")


if __name__ == "__main__":
    main()