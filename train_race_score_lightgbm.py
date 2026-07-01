from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

IN = Path("data/features_rich.csv")
OUT = Path("models/race_score_lgbm_is_over_50.joblib")
PRED_OUT = Path("data/race_score_predictions_valid.csv")
TARGET = "is_over_50"

LIVE_FEATURES = [
    "place",
    "race_no",
    "grade",
    "weather",
    "wind_speed",
    "avg_score",
    "max_score",
    "min_score",
    "std_score",
    "score_gap",
    "avg_age",
    "racer_count",
    "avg_win_rate",
    "max_win_rate",
    "avg_place_rate",
    "max_place_rate",
    "nige_count",
    "oikomi_count",
    "ryo_count",
    "sashi_count",
    "makuri_count",
    "front_runner_pressure",
]

CAT_COLS = ["place", "grade", "weather"]

def make_race_date(df):
    race_id = df["race_id"].astype(str)
    start = pd.to_datetime(race_id.str[2:10], format="%Y%m%d", errors="coerce")
    day_no = pd.to_numeric(race_id.str[10:12], errors="coerce").fillna(1).astype(int)
    return start + pd.to_timedelta(day_no - 1, unit="D")

def prepare(df):
    x = df.copy()

    for col in LIVE_FEATURES:
        if col not in x.columns:
            x[col] = "unknown" if col in CAT_COLS else 0

    x = x[LIVE_FEATURES].copy()

    for col in CAT_COLS:
        if col in x.columns:
            x[col] = x[col].astype("object").fillna("unknown").astype("category")

    for col in x.columns:
        if str(x[col].dtype) != "category":
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    cat_cols = [c for c in CAT_COLS if c in x.columns]
    return x, cat_cols

def main():
    df = pd.read_csv(IN, dtype={"race_id": str}, low_memory=False)
    if TARGET not in df.columns:
        raise SystemExit(f"target column not found: {TARGET}")

    df["race_date"] = make_race_date(df)

    train = df[df["race_date"] < "2026-01-01"].copy()
    valid = df[(df["race_date"] >= "2026-01-01") & (df["race_date"] <= "2026-06-26")].copy()

    x_train, cat_cols = prepare(train)
    x_valid, _ = prepare(valid)
    x_valid = x_valid.reindex(columns=x_train.columns, fill_value=0)

    for col in cat_cols:
        x_valid[col] = x_valid[col].astype("category")

    y_train = train[TARGET].astype(int)
    y_valid = valid[TARGET].astype(int)

    scale_pos_weight = max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum()))

    print(f"input: {IN}")
    print(f"features: {len(x_train.columns)}")
    print(f"train rows: {len(train):,} / positive: {int(y_train.sum()):,} ({y_train.mean():.2%})")
    print(f"valid rows: {len(valid):,} / positive: {int(y_valid.sum()):,} ({y_valid.mean():.2%})")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=48,
        min_child_samples=250,
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
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=cat_cols if cat_cols else "auto",
    )

    pred = model.predict_proba(x_valid)[:, 1]

    print(f"valid AUC: {roc_auc_score(y_valid, pred):.5f}")
    print(f"valid AP : {average_precision_score(y_valid, pred):.5f}")
    print(f"score range: min={pred.min():.4f} max={pred.max():.4f} avg={pred.mean():.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": list(x_train.columns),
            "categorical": cat_cols,
            "target": TARGET,
            "feature_policy": "live_features_only",
        },
        OUT,
    )
    print(f"saved: {OUT}")

    out = valid[["race_id", "race_date", TARGET, "payout_3rentan", "payout_popularity"]].copy()
    out["race_score"] = pred
    out.to_csv(PRED_OUT, index=False, encoding="utf-8-sig")
    print(f"saved: {PRED_OUT}")

if __name__ == "__main__":
    main()
