from pathlib import Path
import sqlite3

import pandas as pd


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "race.db"
FEATURES_PATH = DATA_DIR / "features.csv"


STYLE_COLUMNS = {
    "逃げ": "nige_count",
    "追込": "oikomi_count",
    "両": "ryo_count",
    "差し": "sashi_count",
    "捲り": "makuri_count",
}


def is_rainy(weather: str) -> int:
    return int("雨" in str(weather))


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} が見つかりません。先に make_db.py を実行してください。")

    with sqlite3.connect(DB_PATH) as conn:
        races = pd.read_sql_query("SELECT * FROM races", conn)
        entries = pd.read_sql_query("SELECT * FROM entries", conn)
        payouts = pd.read_sql_query(
            "SELECT race_id, payout AS payout_3rentan, popularity AS payout_popularity "
            "FROM payouts WHERE bet_type = '3連単'",
            conn,
        )

    numeric_cols = ["score", "age", "win_rate", "place_rate"]
    entries[numeric_cols] = entries[numeric_cols].apply(pd.to_numeric, errors="coerce")

    base_features = entries.groupby("race_id").agg(
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        min_score=("score", "min"),
        std_score=("score", "std"),
        avg_age=("age", "mean"),
        racer_count=("car_no", "count"),
        avg_win_rate=("win_rate", "mean"),
        max_win_rate=("win_rate", "max"),
        avg_place_rate=("place_rate", "mean"),
        max_place_rate=("place_rate", "max"),
    )
    base_features["std_score"] = base_features["std_score"].fillna(0)
    base_features["score_gap"] = base_features["max_score"] - base_features["min_score"]

    style_counts = pd.crosstab(entries["race_id"], entries["style"])
    for style, column in STYLE_COLUMNS.items():
        base_features[column] = style_counts.get(style, 0)

    # 逃げ選手が多いほど、先行争いでレースが崩れやすいという仮の特徴量。
    base_features["front_runner_pressure"] = base_features["nige_count"] / base_features["racer_count"]

    features = (
        races.merge(base_features.reset_index(), on="race_id", how="left")
        .merge(payouts, on="race_id", how="left")
        .sort_values("race_id")
    )

    features["wind_speed"] = pd.to_numeric(features["wind_speed"], errors="coerce")
    features["is_rain"] = features["weather"].apply(is_rainy)
    features["is_strong_wind"] = (features["wind_speed"] >= 5.0).astype(int)

    features["payout_3rentan"] = pd.to_numeric(features["payout_3rentan"], errors="coerce")
    features["payout_popularity"] = pd.to_numeric(features["payout_popularity"], errors="coerce")
    features["is_over_100"] = (features["payout_3rentan"] >= 100_000).astype(int)
    features["is_over_300"] = (features["payout_3rentan"] >= 300_000).astype(int)
    features["is_over_500"] = (features["payout_3rentan"] >= 500_000).astype(int)

    ordered_columns = [
        "race_id",
        "date",
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
        "is_rain",
        "is_strong_wind",
        "payout_3rentan",
        "payout_popularity",
        "is_over_100",
        "is_over_300",
        "is_over_500",
    ]

    features = features[ordered_columns]
    features.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")

    print(f"AI用特徴量CSVを作成しました: {FEATURES_PATH}")
    print(features)


if __name__ == "__main__":
    main()
