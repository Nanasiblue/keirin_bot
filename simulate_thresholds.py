from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


DATA_DIR = Path("data")
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

TOP_PCTS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
PROB_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


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


def summarize_selection(df: pd.DataFrame, selected: pd.DataFrame, label: str, target: str) -> dict[str, object]:
    total = len(df)
    count = len(selected)
    payout = pd.to_numeric(selected["payout_3rentan"], errors="coerce") if count else pd.Series(dtype=float)
    return {
        "target": target,
        "rule": label,
        "selected_races": count,
        "selected_rate": count / total if total else 0,
        "avg_score": selected["ai_score"].mean() if count else 0,
        "min_score": selected["ai_score"].min() if count else 0,
        "max_score": selected["ai_score"].max() if count else 0,
        "avg_payout": payout.mean() if count else 0,
        "median_payout": payout.median() if count else 0,
        "max_payout": payout.max() if count else 0,
        "over_30_rate": (selected["payout_3rentan"] >= 30_000).mean() if count else 0,
        "over_50_rate": (selected["payout_3rentan"] >= 50_000).mean() if count else 0,
        "over_100_rate": (selected["payout_3rentan"] >= 100_000).mean() if count else 0,
        "target_hit_rate": selected[target].mean() if count else 0,
    }


def simulate(scored: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    base = {
        "target": target,
        "rule": "ALL",
        "selected_races": len(scored),
        "selected_rate": 1.0,
        "avg_score": scored["ai_score"].mean(),
        "min_score": scored["ai_score"].min(),
        "max_score": scored["ai_score"].max(),
        "avg_payout": scored["payout_3rentan"].mean(),
        "median_payout": scored["payout_3rentan"].median(),
        "max_payout": scored["payout_3rentan"].max(),
        "over_30_rate": (scored["payout_3rentan"] >= 30_000).mean(),
        "over_50_rate": (scored["payout_3rentan"] >= 50_000).mean(),
        "over_100_rate": (scored["payout_3rentan"] >= 100_000).mean(),
        "target_hit_rate": scored[target].mean(),
    }
    rows.append(base)

    sorted_df = scored.sort_values("ai_score", ascending=False)
    for pct in TOP_PCTS:
        n = max(1, int(len(sorted_df) * pct))
        selected = sorted_df.head(n)
        rows.append(summarize_selection(scored, selected, f"top_{int(pct * 100)}pct", target))

    for threshold in PROB_THRESHOLDS:
        selected = scored[scored["ai_score"] >= threshold]
        rows.append(summarize_selection(scored, selected, f"score>={threshold:.2f}", target))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="荒れ予想AIのスコア閾値別シミュレーションをします。")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--target", default="is_over_50", choices=["is_over_30", "is_over_50", "is_over_100"])
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-01")
    parser.add_argument("--valid-end", default="2026-06-26")
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"race_id": str, "race_date_id": str})
    df = add_race_date(df)
    df["payout_3rentan"] = pd.to_numeric(df["payout_3rentan"], errors="coerce")
    df = df.dropna(subset=["race_date", "payout_3rentan"])

    train_end = pd.Timestamp(args.train_end)
    valid_start = pd.Timestamp(args.valid_start)
    valid_end = pd.Timestamp(args.valid_end)

    train_df = df[df["race_date"] <= train_end].copy()
    valid_df = df[(df["race_date"] >= valid_start) & (df["race_date"] <= valid_end)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("学習または検証データが空です。日付指定を確認してください。")

    X_train, y_train = build_xy(train_df, args.target)
    X_valid, y_valid = build_xy(valid_df, args.target, train_columns=list(X_train.columns))

    model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=20,
    )
    model.fit(X_train, y_train)
    valid_df["ai_score"] = model.predict_proba(X_valid)[:, 1]

    auc = roc_auc_score(y_valid, valid_df["ai_score"])
    print(f"target: {args.target}")
    print(f"train: <= {args.train_end} / rows={len(train_df):,} / positive={int(y_train.sum()):,} ({y_train.mean():.2%})")
    print(f"valid: {args.valid_start}..{args.valid_end} / rows={len(valid_df):,} / positive={int(y_valid.sum()):,} ({y_valid.mean():.2%})")
    print(f"valid AUC: {auc:.4f}")

    sim = simulate(valid_df, args.target)
    sim.insert(0, "valid_auc", auc)
    sim_path = DATA_DIR / f"threshold_simulation_{args.target}.csv"
    pred_path = DATA_DIR / f"valid_predictions_{args.target}.csv"
    sim.to_csv(sim_path, index=False, encoding="utf-8-sig")
    valid_df[["race_id", "race_date", "place", "race_no", "ai_score", "payout_3rentan", "payout_popularity", "combination", "is_over_30", "is_over_50", "is_over_100"]].sort_values("ai_score", ascending=False).to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"saved: {sim_path}")
    print(f"saved: {pred_path}")
    print(sim.to_string(index=False, formatters={
        "selected_rate": "{:.2%}".format,
        "avg_score": "{:.4f}".format,
        "min_score": "{:.4f}".format,
        "max_score": "{:.4f}".format,
        "avg_payout": "{:.0f}".format,
        "median_payout": "{:.0f}".format,
        "max_payout": "{:.0f}".format,
        "over_30_rate": "{:.2%}".format,
        "over_50_rate": "{:.2%}".format,
        "over_100_rate": "{:.2%}".format,
        "target_hit_rate": "{:.2%}".format,
    }))


if __name__ == "__main__":
    main()
