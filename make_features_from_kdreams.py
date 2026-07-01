from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
ENTRIES_PATH = DATA_DIR / "entries_from_kdreams.csv"
PAYOUTS_PATH = DATA_DIR / "payouts_from_kdreams.csv"
OUTPUT_PATH = DATA_DIR / "features_from_kdreams.csv"

STYLE_COLUMNS = {
    "逃げ": "nige_count",
    "追込": "oikomi_count",
    "両": "ryo_count",
    "差し": "sashi_count",
    "捲り": "makuri_count",
}


def main() -> None:
    if not ENTRIES_PATH.exists():
        raise FileNotFoundError(f"{ENTRIES_PATH} が見つかりません。先に parse_entries_kdreams.py を実行してください。")
    if not PAYOUTS_PATH.exists():
        raise FileNotFoundError(f"{PAYOUTS_PATH} が見つかりません。先に parse_payouts_kdreams.py を実行してください。")

    entries = pd.read_csv(ENTRIES_PATH)
    payouts = pd.read_csv(PAYOUTS_PATH)

    numeric_cols = ["score", "age", "win_rate", "place_rate"]
    entries[numeric_cols] = entries[numeric_cols].apply(pd.to_numeric, errors="coerce")

    features = entries.groupby("race_id").agg(
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
    features["std_score"] = features["std_score"].fillna(0)
    features["score_gap"] = features["max_score"] - features["min_score"]

    style_counts = pd.crosstab(entries["race_id"], entries["style"])
    for style, column in STYLE_COLUMNS.items():
        features[column] = style_counts.get(style, 0)
    features["front_runner_pressure"] = features["nige_count"] / features["racer_count"]

    payouts = payouts[payouts["bet_type"] == "3連単"].copy()
    payouts["payout"] = pd.to_numeric(payouts["payout"], errors="coerce")
    payouts["popularity"] = pd.to_numeric(payouts["popularity"], errors="coerce")
    payouts = payouts.rename(columns={"payout": "payout_3rentan", "popularity": "payout_popularity"})

    output = features.reset_index().merge(
        payouts[["race_id", "combination", "payout_3rentan", "payout_popularity"]],
        on="race_id",
        how="inner",
    )

    output["is_over_30"] = (output["payout_3rentan"] >= 30_000).astype(int)
    output["is_over_50"] = (output["payout_3rentan"] >= 50_000).astype(int)
    output["is_over_100"] = (output["payout_3rentan"] >= 100_000).astype(int)
    output["is_popular_50plus"] = (output["payout_popularity"] >= 50).astype(int)
    output["is_popular_100plus"] = (output["payout_popularity"] >= 100).astype(int)
    output["is_popular_150plus"] = (output["payout_popularity"] >= 150).astype(int)

    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"実データ特徴量CSVを作成しました: {OUTPUT_PATH}")
    print(output)
    if output.empty:
        print("注意: entries と payouts の race_id が一致していません。probe_kdreams.py を初日URLで再実行してください。")


if __name__ == "__main__":
    main()
