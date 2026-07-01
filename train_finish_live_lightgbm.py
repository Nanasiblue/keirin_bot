from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

IN = Path("data/race_results_full_clean_v2.csv")
MODEL_DIR = Path("models")
PRED_OUT = Path("data/finish_live_lightgbm_predictions_valid.csv")

LIVE_FEATURES = [
    "car_no",
    "score",
    "age",
    "win_rate",
    "place_rate",
    "style",
    "score_rank_in_race",
    "win_rate_rank_in_race",
    "place_rate_rank_in_race",
    "racer_count",
    "avg_score",
    "max_score",
    "min_score",
    "std_score",
    "score_gap",
    "avg_age",
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

CAT_COLS = ["style"]

TARGETS = {
    "p_1st": "ticket_1st",
    "p_top2": "ticket_top2",
    "p_top3": "ticket_top3",
}


def make_race_date(df: pd.DataFrame) -> pd.Series:
    race_id = df["race_id"].astype(str)
    start = pd.to_datetime(race_id.str[2:10], format="%Y%m%d", errors="coerce")
    day_no = pd.to_numeric(race_id.str[10:12], errors="coerce").fillna(1).astype(int)
    return start + pd.to_timedelta(day_no - 1, unit="D")


def add_live_race_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["score", "age", "win_rate", "place_rate", "car_no"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["score_rank_in_race"] = out.groupby("race_id")["score"].rank(ascending=False, method="min")
    out["win_rate_rank_in_race"] = out.groupby("race_id")["win_rate"].rank(ascending=False, method="min")
    out["place_rate_rank_in_race"] = out.groupby("race_id")["place_rate"].rank(ascending=False, method="min")

    style = out["style"].fillna("").astype(str)
    out["_nige"] = style.str.contains("逃").astype(int)
    out["_oikomi"] = style.str.contains("追").astype(int)
    out["_ryo"] = style.str.contains("両").astype(int)
    out["_sashi"] = style.str.contains("差").astype(int)
    out["_makuri"] = style.str.contains("捲").astype(int)

    g = out.groupby("race_id")

    out["racer_count"] = g["car_no"].transform("count")
    out["avg_score"] = g["score"].transform("mean")
    out["max_score"] = g["score"].transform("max")
    out["min_score"] = g["score"].transform("min")
    out["std_score"] = g["score"].transform(lambda s: s.std(ddof=0))
    out["score_gap"] = out["max_score"] - out["min_score"]

    out["avg_age"] = g["age"].transform("mean")
    out["avg_win_rate"] = g["win_rate"].transform("mean")
    out["max_win_rate"] = g["win_rate"].transform("max")
    out["avg_place_rate"] = g["place_rate"].transform("mean")
    out["max_place_rate"] = g["place_rate"].transform("max")

    out["nige_count"] = g["_nige"].transform("sum")
    out["oikomi_count"] = g["_oikomi"].transform("sum")
    out["ryo_count"] = g["_ryo"].transform("sum")
    out["sashi_count"] = g["_sashi"].transform("sum")
    out["makuri_count"] = g["_makuri"].transform("sum")
    out["front_runner_pressure"] = out["nige_count"] / out["racer_count"].replace(0, 1)

    return out


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = df.copy()

    for col in LIVE_FEATURES:
        if col not in x.columns:
            x[col] = "unknown" if col in CAT_COLS else 0

    x = x[LIVE_FEATURES].copy()

    for col in CAT_COLS:
        x[col] = x[col].astype("object").fillna("unknown").astype("category")

    for col in x.columns:
        if str(x[col].dtype) != "category":
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    cat_cols = [c for c in CAT_COLS if c in x.columns]
    return x, cat_cols


def train_one(df: pd.DataFrame, target_name: str, target_col: str) -> pd.DataFrame:
    train = df[df["race_date"] < "2026-01-01"].copy()
    valid = df[(df["race_date"] >= "2026-01-01") & (df["race_date"] <= "2026-06-26")].copy()

    x_train, cat_cols = prepare(train)
    x_valid, _ = prepare(valid)
    x_valid = x_valid.reindex(columns=x_train.columns, fill_value=0)

    for col in cat_cols:
        x_valid[col] = x_valid[col].astype("category")

    y_train = train[target_col].astype(int)
    y_valid = valid[target_col].astype(int)

    scale_pos_weight = max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum()))

    print("")
    print(f"target: {target_name}")
    print(f"features: {len(x_train.columns)}")
    print(f"train positive: {int(y_train.sum()):,} / {len(y_train):,} ({y_train.mean():.2%})")
    print(f"valid positive: {int(y_valid.sum()):,} / {len(y_valid):,} ({y_valid.mean():.2%})")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.035,
        num_leaves=48,
        min_child_samples=200,
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

    print(f"AUC: {roc_auc_score(y_valid, pred):.5f}")
    print(f"AP : {average_precision_score(y_valid, pred):.5f}")
    print(f"score range: min={pred.min():.4f} max={pred.max():.4f} avg={pred.mean():.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"finish_live_lgbm_{target_name}.joblib"
    joblib.dump(
        {
            "model": model,
            "features": list(x_train.columns),
            "categorical": cat_cols,
            "target": target_col,
            "feature_policy": "live_features_only",
        },
        model_path,
    )
    print(f"saved: {model_path}")

    out = valid[["race_id", "car_no", "name", "race_date"]].copy()
    out[target_name] = pred
    return out


def main() -> None:
    df = pd.read_csv(IN, dtype={"race_id": str}, low_memory=False)

    required = ["race_id", "car_no", "score", "style", "age", "win_rate", "place_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    for col in TARGETS.values():
        if col not in df.columns:
            raise SystemExit(f"missing target column: {col}")

    df["race_date"] = make_race_date(df)
    df = add_live_race_features(df)

    preds = None
    for target_name, target_col in TARGETS.items():
        pred = train_one(df, target_name, target_col)
        if preds is None:
            preds = pred
        else:
            preds = preds.merge(
                pred[["race_id", "car_no", target_name]],
                on=["race_id", "car_no"],
                how="left",
            )

    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PRED_OUT, index=False, encoding="utf-8-sig")
    print("")
    print(f"saved: {PRED_OUT}")


if __name__ == "__main__":
    main()
