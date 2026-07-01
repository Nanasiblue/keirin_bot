from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DEFAULT_INPUT = DATA_DIR / "features_rich.csv"
DROP_COLS = {"race_id", "race_date_id", "combination", "payout_3rentan", "payout_popularity", "is_over_30", "is_over_50", "is_over_100", "is_popular_50plus", "is_popular_100plus", "is_popular_150plus", "parse_error"}


def add_race_date(df: pd.DataFrame) -> pd.DataFrame:
    copied = df.copy()
    race_id = copied["race_id"].astype(str).str.zfill(16)
    start_date = pd.to_datetime(race_id.str[2:10], format="%Y%m%d", errors="coerce")
    day_no = pd.to_numeric(race_id.str[10:12], errors="coerce").fillna(1).astype(int)
    copied["race_start_date"] = start_date
    copied["race_day_no"] = day_no
    copied["race_date"] = start_date + pd.to_timedelta(day_no - 1, unit="D")
    return copied


def build_xy(df: pd.DataFrame, target: str, train_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target].astype(int)
    feature_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()
    feature_df = feature_df.drop(columns=["race_date", "race_start_date"], errors="ignore")
    categorical_cols = [c for c in ["place", "weather"] if c in feature_df.columns]
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=True)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    if train_columns is not None:
        feature_df = feature_df.reindex(columns=train_columns, fill_value=0)
    return feature_df, y


def candidate_params(trees: int) -> list[dict[str, object]]:
    return [
        {"n_estimators": trees, "min_samples_leaf": 10, "max_features": "sqrt", "max_depth": None},
        {"n_estimators": trees, "min_samples_leaf": 20, "max_features": "sqrt", "max_depth": None},
        {"n_estimators": trees, "min_samples_leaf": 40, "max_features": "sqrt", "max_depth": None},
        {"n_estimators": trees, "min_samples_leaf": 20, "max_features": 0.5, "max_depth": None},
        {"n_estimators": trees, "min_samples_leaf": 40, "max_features": 0.5, "max_depth": None},
        {"n_estimators": trees, "min_samples_leaf": 20, "max_features": "log2", "max_depth": None},
        {"n_estimators": trees, "min_samples_leaf": 20, "max_features": "sqrt", "max_depth": 12},
        {"n_estimators": trees, "min_samples_leaf": 40, "max_features": "sqrt", "max_depth": 12},
    ]


def score_rows(scored: pd.DataFrame, target: str, model_name: str, auc: float, params: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    sorted_df = scored.sort_values("ai_score", ascending=False)
    rules = [("ALL", scored)]
    for pct in [0.01, 0.03, 0.05, 0.10]:
        rules.append((f"top_{int(pct * 100)}pct", sorted_df.head(max(1, int(len(sorted_df) * pct)))))
    for threshold in [0.35, 0.40, 0.45, 0.50]:
        rules.append((f"score>={threshold:.2f}", scored[scored["ai_score"] >= threshold]))
    for rule, df in rules:
        count = len(df)
        rows.append({
            "model": model_name,
            "params": str(params),
            "auc": auc,
            "rule": rule,
            "selected_races": count,
            "selected_rate": count / len(scored) if len(scored) else 0,
            "avg_score": df["ai_score"].mean() if count else 0,
            "avg_payout": df["payout_3rentan"].mean() if count else 0,
            "median_payout": df["payout_3rentan"].median() if count else 0,
            "max_payout": df["payout_3rentan"].max() if count else 0,
            "over_30_rate": (df["payout_3rentan"] >= 30_000).mean() if count else 0,
            "over_50_rate": (df["payout_3rentan"] >= 50_000).mean() if count else 0,
            "over_100_rate": (df["payout_3rentan"] >= 100_000).mean() if count else 0,
            "target_hit_rate": df[target].mean() if count else 0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="RandomForestを荒れレースフィルタ向けにチューニングします。")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--target", default="is_over_50", choices=["is_over_30", "is_over_50", "is_over_100"])
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-01")
    parser.add_argument("--valid-end", default="2026-06-26")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"race_id": str, "race_date_id": str})
    df = add_race_date(df)
    df["payout_3rentan"] = pd.to_numeric(df["payout_3rentan"], errors="coerce")
    df = df.dropna(subset=["race_date", "payout_3rentan"])
    train_df = df[df["race_date"] <= pd.Timestamp(args.train_end)].copy()
    valid_df = df[(df["race_date"] >= pd.Timestamp(args.valid_start)) & (df["race_date"] <= pd.Timestamp(args.valid_end))].copy()
    X_train, y_train = build_xy(train_df, args.target)
    X_valid, y_valid = build_xy(valid_df, args.target, train_columns=list(X_train.columns))
    print(f"target: {args.target}")
    print(f"train rows: {len(train_df):,} positive: {int(y_train.sum()):,} ({y_train.mean():.2%})")
    print(f"valid rows: {len(valid_df):,} positive: {int(y_valid.sum()):,} ({y_valid.mean():.2%})")
    print(f"features: {len(X_train.columns):,}")

    all_rows = []
    best_model = None
    best_params = None
    best_key = (-1.0, -1.0, -1.0)
    for i, params in enumerate(candidate_params(args.trees), start=1):
        print("=" * 80)
        print(f"[{i}] {params}")
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight="balanced", bootstrap=True)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, proba)
        scored = valid_df.copy()
        scored["ai_score"] = proba
        rows = score_rows(scored, args.target, f"rf_{i}", auc, params)
        all_rows.extend(rows)
        summary = pd.DataFrame(rows)
        print(f"AUC: {auc:.4f}")
        focus = summary[summary["rule"].isin(["top_3pct", "top_5pct", "score>=0.40", "score>=0.50"])]
        print(focus.to_string(index=False, formatters={"selected_rate":"{:.2%}".format,"avg_score":"{:.4f}".format,"avg_payout":"{:.0f}".format,"median_payout":"{:.0f}".format,"max_payout":"{:.0f}".format,"over_30_rate":"{:.2%}".format,"over_50_rate":"{:.2%}".format,"over_100_rate":"{:.2%}".format,"target_hit_rate":"{:.2%}".format}))
        top3 = summary[summary["rule"] == "top_3pct"].iloc[0]
        top5 = summary[summary["rule"] == "top_5pct"].iloc[0]
        key = (float(top3["target_hit_rate"]), float(top5["target_hit_rate"]), auc)
        if key > best_key:
            best_key = key
            best_model = model
            best_params = params

    result = pd.DataFrame(all_rows)
    out_path = DATA_DIR / f"random_forest_tuning_{args.target}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("=" * 80)
    print(f"saved: {out_path}")
    print(f"best_key(top3, top5, auc): {best_key}")
    print(f"best_params: {best_params}")
    if args.save_best and best_model is not None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"random_forest_tuned_{args.target}.joblib"
        joblib.dump({"model": best_model, "target": args.target, "columns": list(X_train.columns), "params": best_params}, model_path)
        print(f"saved: {model_path}")

if __name__ == "__main__":
    main()
