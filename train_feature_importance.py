from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data")
DEFAULT_INPUT = DATA_DIR / "features_rich.csv"
FALLBACK_INPUT = DATA_DIR / "features_all_kdreams.csv"

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


def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"race_id": str, "race_date_id": str})
    return pd.read_csv(FALLBACK_INPUT, dtype={"race_id": str})


def build_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target].astype(int)
    feature_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()
    categorical_cols = [c for c in ["place", "weather"] if c in feature_df.columns]
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=True)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return feature_df, y


def main() -> None:
    parser = argparse.ArgumentParser(description="荒れ予想AIの特徴量重要度を出します。")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--target", default="is_over_30", choices=["is_over_30", "is_over_50", "is_over_100"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    df = load_data(Path(args.input))
    if args.target not in df.columns:
        raise ValueError(f"target列がありません: {args.target}")

    X, y = build_xy(df, args.target)
    positive = int(y.sum())
    print(f"input: {args.input}")
    print(f"rows: {len(df):,}")
    print(f"features: {len(X.columns):,}")
    print(f"target: {args.target}")
    print(f"positive: {positive:,} ({positive / len(y):.2%})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=20,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, pred, digits=4))

    importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    out_path = DATA_DIR / f"feature_importance_{args.target}.csv"
    importance.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved: {out_path}")
    print(importance.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
