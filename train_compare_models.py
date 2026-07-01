from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DEFAULT_INPUT = DATA_DIR / "features_rich.csv"
DROP_COLS = {
    "race_id",
    "race_date_id",
    "combination",
    "payout_3rentan",
    "payout_popularity",
    "is_over_30",
    "is_over_50",
    "is_over_100",
    "is_popular_50plus",
    "is_popular_100plus",
    "is_popular_150plus",
    "parse_error",
}


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


def make_models(trees: int) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=trees,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=20,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=trees,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=20,
        ),
    }
    models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as exc:
        print(f"LightGBMは使いません: {exc}")

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=20,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    except Exception as exc:
        print(f"XGBoostは使いません: {exc}")

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as exc:
        print(f"CatBoostは使いません: {exc}")

    return models


def summarize(scored: pd.DataFrame, target: str, model_name: str, auc: float) -> list[dict[str, object]]:
    rows = []
    rules = [("ALL", scored)]
    sorted_df = scored.sort_values("ai_score", ascending=False)
    for pct in [0.01, 0.03, 0.05, 0.10, 0.20]:
        n = max(1, int(len(sorted_df) * pct))
        rules.append((f"top_{int(pct * 100)}pct", sorted_df.head(n)))
    for threshold in [0.30, 0.40, 0.50, 0.60]:
        rules.append((f"score>={threshold:.2f}", scored[scored["ai_score"] >= threshold]))

    for rule, df in rules:
        count = len(df)
        rows.append({
            "model": model_name,
            "target": target,
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
    parser = argparse.ArgumentParser(description="荒れ予想AIのモデル比較をします。")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--target", default="is_over_50", choices=["is_over_30", "is_over_50", "is_over_100"])
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-01")
    parser.add_argument("--valid-end", default="2026-06-26")
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"race_id": str, "race_date_id": str})
    df = add_race_date(df)
    df["payout_3rentan"] = pd.to_numeric(df["payout_3rentan"], errors="coerce")
    df = df.dropna(subset=["race_date", "payout_3rentan"])

    train_df = df[df["race_date"] <= pd.Timestamp(args.train_end)].copy()
    valid_df = df[(df["race_date"] >= pd.Timestamp(args.valid_start)) & (df["race_date"] <= pd.Timestamp(args.valid_end))].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("学習または検証データが空です。日付指定を確認してください。")

    X_train, y_train = build_xy(train_df, args.target)
    X_valid, y_valid = build_xy(valid_df, args.target, train_columns=list(X_train.columns))
    print(f"target: {args.target}")
    print(f"train rows: {len(train_df):,} positive: {int(y_train.sum()):,} ({y_train.mean():.2%})")
    print(f"valid rows: {len(valid_df):,} positive: {int(y_valid.sum()):,} ({y_valid.mean():.2%})")
    print(f"features: {len(X_train.columns):,}")

    all_rows = []
    best_auc = -1.0
    best_name = None
    best_model = None
    for name, model in make_models(args.trees).items():
        print("=" * 80)
        print(f"fit: {name}")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, proba)
        print(f"AUC: {auc:.4f}")
        scored = valid_df.copy()
        scored["ai_score"] = proba
        rows = summarize(scored, args.target, name, auc)
        all_rows.extend(rows)
        print(pd.DataFrame(rows).head(8).to_string(index=False, formatters={
            "selected_rate": "{:.2%}".format,
            "avg_score": "{:.4f}".format,
            "avg_payout": "{:.0f}".format,
            "median_payout": "{:.0f}".format,
            "max_payout": "{:.0f}".format,
            "over_30_rate": "{:.2%}".format,
            "over_50_rate": "{:.2%}".format,
            "over_100_rate": "{:.2%}".format,
            "target_hit_rate": "{:.2%}".format,
        }))
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    result = pd.DataFrame(all_rows)
    out_path = DATA_DIR / f"model_compare_{args.target}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("=" * 80)
    print(f"saved: {out_path}")
    print(f"best_auc: {best_name} {best_auc:.4f}")

    if args.save_best and best_model is not None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"best_model_{args.target}.joblib"
        columns_path = MODEL_DIR / f"best_model_{args.target}_columns.csv"
        joblib.dump({"model": best_model, "target": args.target, "columns": list(X_train.columns), "model_name": best_name}, model_path)
        pd.Series(X_train.columns, name="feature").to_csv(columns_path, index=False, encoding="utf-8-sig")
        print(f"saved: {model_path}")
        print(f"saved: {columns_path}")


if __name__ == "__main__":
    main()
