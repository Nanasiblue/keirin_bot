from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

IN = Path("data/direct_ticket_dataset.csv")
MODEL_OUT = Path("models/direct_ticket_lightgbm.joblib")
PRED_OUT = Path("data/direct_ticket_predictions_valid.csv")
IMP_OUT = Path("data/direct_ticket_feature_importance.csv")

DROP_COLS = {
    "race_id", "combination", "hit_combination", "hit_payout", "hit_popularity",
    "is_hit", "race_date",
    "payout_3rentan", "payout_popularity", "is_over_30", "is_over_50", "is_over_100",
}

CATEGORICAL_COLS = ["first_style", "second_style", "third_style", "place", "weather"]

KEEP_ID_COLS = [
    "race_id", "combination", "odds", "estimated_payout",
    "is_hit", "hit_payout", "hit_popularity", "race_date",
]


def make_race_date(df):
    if "race_date" in df.columns:
        return pd.to_datetime(df["race_date"], errors="coerce")

    race_id = df["race_id"].astype(str)
    start = pd.to_datetime(race_id.str[2:10], format="%Y%m%d", errors="coerce")
    day_no = pd.to_numeric(race_id.str[10:12], errors="coerce").fillna(1).astype(int)
    return start + pd.to_timedelta(day_no - 1, unit="D")


def prepare_features(df):
    y = df["is_hit"].astype(int)

    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop).copy()

    for col in list(X.columns):
        if X[col].dtype == "object":
            if col in CATEGORICAL_COLS:
                X[col] = X[col].astype("object").fillna("unknown").astype("category")
            else:
                X = X.drop(columns=[col])

    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("object").fillna("unknown").astype("category")

    for col in X.columns:
        if str(X[col].dtype) != "category":
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    return X, y, cat_cols


def main():
    print(f"input: {IN}")
    df = pd.read_csv(IN, dtype={"race_id": str, "combination": str}, low_memory=False)
    df["race_date"] = make_race_date(df)

    print(f"rows: {len(df):,}")
    print(f"races: {df['race_id'].nunique():,}")
    print(f"positive: {int(df['is_hit'].sum()):,} ({df['is_hit'].mean():.4%})")

    train_mask = df["race_date"] < pd.Timestamp("2026-01-01")
    valid_mask = (df["race_date"] >= pd.Timestamp("2026-01-01")) & (df["race_date"] <= pd.Timestamp("2026-06-26"))

    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()

    print(f"train rows: {len(train_df):,} / races: {train_df['race_id'].nunique():,} / pos: {int(train_df['is_hit'].sum()):,} ({train_df['is_hit'].mean():.4%})")
    print(f"valid rows: {len(valid_df):,} / races: {valid_df['race_id'].nunique():,} / pos: {int(valid_df['is_hit'].sum()):,} ({valid_df['is_hit'].mean():.4%})")

    X_train, y_train, cat_cols = prepare_features(train_df)
    X_valid, y_valid, _ = prepare_features(valid_df)
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

    for col in cat_cols:
        if col in X_valid.columns:
            X_valid[col] = X_valid[col].astype("category")

    print(f"features: {X_train.shape[1]}")
    print(f"categorical: {cat_cols}")

    scale_pos_weight = max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=96,
        min_child_samples=300,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=cat_cols if cat_cols else "auto",
    )

    pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, pred)
    ap = average_precision_score(y_valid, pred)

    print(f"valid AUC: {auc:.5f}")
    print(f"valid AP : {ap:.5f}")

    threshold = np.quantile(pred, 0.99)
    pred_label = (pred >= threshold).astype(int)
    print(f"top1pct threshold: {threshold:.6f}")
    print(classification_report(y_valid, pred_label, digits=4))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(X_train.columns), "categorical": cat_cols},
        MODEL_OUT,
    )
    print(f"saved: {MODEL_OUT}")

    out_cols = [c for c in KEEP_ID_COLS if c in valid_df.columns]
    pred_df = valid_df[out_cols].copy()
    pred_df["direct_ticket_score"] = pred
    pred_df["direct_ticket_rank"] = pred_df.groupby("race_id")["direct_ticket_score"].rank(
        ascending=False,
        method="first",
    ).astype(int)
    pred_df["direct_expected_return"] = pred_df["direct_ticket_score"] * pd.to_numeric(
        pred_df["estimated_payout"],
        errors="coerce",
    ).fillna(0)

    pred_df.to_csv(PRED_OUT, index=False, encoding="utf-8-sig")
    print(f"saved: {PRED_OUT}")

    imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    imp.to_csv(IMP_OUT, index=False, encoding="utf-8-sig")
    print(f"saved: {IMP_OUT}")
    print(imp.head(40).to_string(index=False))


if __name__ == "__main__":
    main()

