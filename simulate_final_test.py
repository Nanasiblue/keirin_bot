from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("data")
DEFAULT_INPUT = DATA_DIR / "features_rich.csv"
DROP_COLS = {"race_id", "race_date_id", "combination", "payout_3rentan", "payout_popularity", "is_over_30", "is_over_50", "is_over_100", "is_popular_50plus", "is_popular_100plus", "is_popular_150plus", "parse_error"}


def add_race_date(df: pd.DataFrame) -> pd.DataFrame:
    copied = df.copy()
    race_id = copied["race_id"].astype(str).str.zfill(16)
    copied["race_date"] = pd.to_datetime(race_id.str[2:10], format="%Y%m%d", errors="coerce")
    return copied


def build_xy(df: pd.DataFrame, target: str, train_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target].astype(int)
    feature_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()
    feature_df = feature_df.drop(columns=["race_date"], errors="ignore")
    categorical_cols = [c for c in ["place", "weather"] if c in feature_df.columns]
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=True)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    if train_columns is not None:
        feature_df = feature_df.reindex(columns=train_columns, fill_value=0)
    return feature_df, y


def summarize(name: str, df: pd.DataFrame, target: str) -> dict[str, object]:
    count = len(df)
    return {
        "rule": name,
        "selected_races": count,
        "avg_score": df["ai_score"].mean() if count else 0,
        "min_score": df["ai_score"].min() if count else 0,
        "max_score": df["ai_score"].max() if count else 0,
        "avg_payout": df["payout_3rentan"].mean() if count else 0,
        "median_payout": df["payout_3rentan"].median() if count else 0,
        "max_payout": df["payout_3rentan"].max() if count else 0,
        "over_30_rate": (df["payout_3rentan"] >= 30_000).mean() if count else 0,
        "over_50_rate": (df["payout_3rentan"] >= 50_000).mean() if count else 0,
        "over_100_rate": (df["payout_3rentan"] >= 100_000).mean() if count else 0,
        "target_hit_rate": df[target].mean() if count else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="6/28以降などの完全未使用データで固定閾値テストをします。")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--target", default="is_over_50", choices=["is_over_30", "is_over_50", "is_over_100"])
    parser.add_argument("--train-end", default="2026-06-26")
    parser.add_argument("--test-start", default="2026-06-28")
    parser.add_argument("--test-end", default="2099-12-31")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"race_id": str, "race_date_id": str})
    df = add_race_date(df)
    df["payout_3rentan"] = pd.to_numeric(df["payout_3rentan"], errors="coerce")
    df = df.dropna(subset=["race_date", "payout_3rentan"])

    train_df = df[df["race_date"] <= pd.Timestamp(args.train_end)].copy()
    test_df = df[(df["race_date"] >= pd.Timestamp(args.test_start)) & (df["race_date"] <= pd.Timestamp(args.test_end))].copy()
    if train_df.empty:
        raise ValueError("学習データが空です。")
    if test_df.empty:
        raise ValueError("テストデータが空です。6/28以降の取得と features_rich.csv の作り直しを確認してください。")

    X_train, y_train = build_xy(train_df, args.target)
    X_test, y_test = build_xy(test_df, args.target, train_columns=list(X_train.columns))
    model = RandomForestClassifier(n_estimators=args.trees, random_state=42, n_jobs=-1, class_weight="balanced", min_samples_leaf=20)
    model.fit(X_train, y_train)
    test_df["ai_score"] = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, test_df["ai_score"]) if y_test.nunique() > 1 else None
    selected = test_df[test_df["ai_score"] >= args.threshold].copy()
    top5 = test_df.sort_values("ai_score", ascending=False).head(max(1, int(len(test_df) * 0.05))).copy()

    result = pd.DataFrame([
        summarize("ALL", test_df, args.target),
        summarize(f"score>={args.threshold:.2f}", selected, args.target),
        summarize("top_5pct", top5, args.target),
    ])
    result.insert(0, "target", args.target)
    result.insert(1, "test_auc", auc if auc is not None else "NA")

    result_path = DATA_DIR / f"final_test_summary_{args.target}.csv"
    pred_path = DATA_DIR / f"final_test_predictions_{args.target}.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    test_df[["race_id", "race_date", "place", "race_no", "ai_score", "payout_3rentan", "payout_popularity", "combination", "is_over_30", "is_over_50", "is_over_100"]].sort_values("ai_score", ascending=False).to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"target: {args.target}")
    print(f"train: <= {args.train_end} / rows={len(train_df):,} / positive={int(y_train.sum()):,} ({y_train.mean():.2%})")
    print(f"test : {args.test_start}..{args.test_end} / rows={len(test_df):,} / positive={int(y_test.sum()):,} ({y_test.mean():.2%})")
    print(f"test AUC: {auc if auc is not None else 'NA'}")
    print(f"saved: {result_path}")
    print(f"saved: {pred_path}")
    print(result.to_string(index=False, formatters={"avg_score":"{:.4f}".format,"min_score":"{:.4f}".format,"max_score":"{:.4f}".format,"avg_payout":"{:.0f}".format,"median_payout":"{:.0f}".format,"max_payout":"{:.0f}".format,"over_30_rate":"{:.2%}".format,"over_50_rate":"{:.2%}".format,"over_100_rate":"{:.2%}".format,"target_hit_rate":"{:.2%}".format}))

if __name__ == "__main__":
    main()
