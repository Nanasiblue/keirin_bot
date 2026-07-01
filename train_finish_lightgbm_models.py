from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

IN = Path("data/racer_rank_dataset.csv")
PRED_OUT = Path("data/finish_lightgbm_predictions_valid.csv")
IMP_OUT = Path("data/finish_lightgbm_feature_importance.csv")
MODEL_DIR = Path("models")

DROP_COLS = {
    "race_id", "race_date", "race_start_date", "date", "name",
    "finish_position", "finish_class",
    "margin", "agari", "decision", "sb", "comment",
    "combination", "payout", "popularity",
    "ticket_1st", "ticket_2nd", "ticket_3rd", "ticket_top2", "ticket_top3",
    "official_is_1st", "official_top2", "official_top3",
    "ticket_1st_car", "ticket_2nd_car", "ticket_3rd_car",
    "has_numeric_finish", "is_finished",
    "finish_status", "finish_status_detail", "same_finish_count", "raw_finish_position",
    "exception_name_from_html", "exception_margin", "exception_agari",
    "exception_decision", "exception_sb", "exception_comment",
    "is_dead_heat", "is_absent", "is_dnf", "is_crash", "is_disqualified",
    "is_accident_finish", "is_unknown_exception",
}

POST_RACE_PATTERNS = [
    "finish", "ticket", "official", "payout", "popularity", "combination",
    "result", "decision", "margin", "agari", "comment", "exception",
    "disqualified", "dead_heat", "dnf", "crash",
]

ALLOW_FEATURE_COLS = {
    "score_rank_in_race",
    "win_rate_rank_in_race",
    "place_rate_rank_in_race",
}

TARGETS = [
    ("1st", "finish_lgbm_1st.joblib", "p_1st"),
    ("top2", "finish_lgbm_top2.joblib", "p_top2"),
    ("top3", "finish_lgbm_top3.joblib", "p_top3"),
]


def is_post_race_col(col):
    if col in ALLOW_FEATURE_COLS:
        return False
    low = col.lower()
    return any(p in low for p in POST_RACE_PATTERNS)


def make_x(df, columns=None):
    drop = set(DROP_COLS)
    drop.update(c for c in df.columns if is_post_race_col(c))
    x = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore").copy()

    cats = [c for c in x.columns if x[c].dtype == "object"]
    x = pd.get_dummies(x, columns=cats, dummy_na=True)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)

    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0)
    return x


def make_y(df, target):
    cls = df["finish_class"].astype(str)
    if target == "1st":
        return (cls == "1st").astype(int)
    if target == "top2":
        return cls.isin(["1st", "2nd"]).astype(int)
    if target == "top3":
        return cls.isin(["1st", "2nd", "3rd"]).astype(int)
    raise ValueError(target)


def train_one(target, X_train, y_train, X_valid, y_valid):
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = max(1.0, neg / max(1, pos))

    print("")
    print(f"=== target: {target} ===")
    print(f"train positive: {pos:,} ({y_train.mean():.4%})")
    print(f"valid positive: {int(y_valid.sum()):,} ({y_valid.mean():.4%})")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=180,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc")

    pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, pred)
    ap = average_precision_score(y_valid, pred)

    print(f"valid AUC: {auc:.5f}")
    print(f"valid AP : {ap:.5f}")

    th = np.quantile(pred, 0.80)
    label = (pred >= th).astype(int)
    print(f"top20pct threshold: {th:.6f}")
    print(classification_report(y_valid, label, digits=4))

    return model, pred, auc, ap


def main():
    print(f"input: {IN}")
    df = pd.read_csv(IN, dtype={"race_id": str}, low_memory=False)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    df = df[df["finish_class"].astype(str) != "absent"].copy()
    df = df.dropna(subset=["race_date", "finish_class", "race_id", "car_no"]).copy()

    train = df[df["race_date"] <= "2025-12-31"].copy()
    valid = df[(df["race_date"] >= "2026-01-01") & (df["race_date"] <= "2026-06-26")].copy()

    print(f"rows: {len(df):,}")
    print(f"train rows: {len(train):,} / races: {train['race_id'].nunique():,}")
    print(f"valid rows: {len(valid):,} / races: {valid['race_id'].nunique():,}")

    X_train = make_x(train)
    X_valid = make_x(valid, list(X_train.columns))
    print(f"features: {len(X_train.columns):,}")

    out_cols = [c for c in ["race_id", "race_date", "car_no", "name", "finish_class"] if c in valid.columns]
    pred_df = valid[out_cols].copy()
    pred_df["race_date"] = pd.to_datetime(pred_df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    imps = []
    metrics = []

    for target, model_name, pred_col in TARGETS:
        y_train = make_y(train, target)
        y_valid = make_y(valid, target)

        model, pred, auc, ap = train_one(target, X_train, y_train, X_valid, y_valid)

        pred_df[pred_col] = pred
        pred_df[f"rank_{pred_col[2:]}"] = pred_df.groupby("race_id")[pred_col].rank(
            ascending=False,
            method="first",
        ).astype(int)

        model_path = MODEL_DIR / model_name
        joblib.dump(
            {
                "model": model,
                "features": list(X_train.columns),
                "target": target,
                "pred_col": pred_col,
            },
            model_path,
            compress=3,
        )

        print(f"saved: {model_path}")
        print(f"model size MB: {model_path.stat().st_size / 1024 / 1024:.2f}")

        imps.append(pd.DataFrame({
            "target": target,
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }))
        metrics.append({"target": target, "auc": auc, "ap": ap, "model": str(model_path)})

    pred_df["finish_ai_summary_rank"] = pred_df.groupby("race_id")["p_top3"].rank(
        ascending=False,
        method="first",
    ).astype(int)

    pred_df.to_csv(PRED_OUT, index=False, encoding="utf-8-sig")
    pd.concat(imps, ignore_index=True).to_csv(IMP_OUT, index=False, encoding="utf-8-sig")

    print("")
    print(f"saved: {PRED_OUT}")
    print(f"saved: {IMP_OUT}")

    print("")
    print("=== metrics ===")
    print(pd.DataFrame(metrics).to_string(index=False))

    print("")
    print("=== sample prediction ===")
    print(
        pred_df.sort_values(["race_date", "race_id", "p_top3"], ascending=[True, True, False])
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
