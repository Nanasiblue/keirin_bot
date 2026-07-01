from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features_all_kdreams.csv"
ENTRIES_PATH = DATA_DIR / "entries_all_kdreams.csv"
PAYOUTS_PATH = DATA_DIR / "payouts_all_kdreams.csv"
DAYS_PATH = DATA_DIR / "kdreams_days_list_2023_20260626_unique.csv"
LOG_PATH = DATA_DIR / "fetch_logs_kdreams.csv"
DAY_ROOT = DATA_DIR / "kdreams_days"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"NG: ファイルがありません: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"race_id": str, "race_date_id": str})


def show_basic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = read_csv(FEATURES_PATH)
    entries = read_csv(ENTRIES_PATH)
    payouts = read_csv(PAYOUTS_PATH)

    print("=== 基本件数 ===")
    print(f"features: {len(features):,} races")
    print(f"entries : {len(entries):,} rows")
    print(f"payouts : {len(payouts):,} rows")
    if not features.empty:
        print(f"race_id重複: {features['race_id'].duplicated().sum():,}")
    print()
    return features, entries, payouts


def show_fetch_progress() -> None:
    print("=== 取得進捗 ===")
    days = read_csv(DAYS_PATH)
    logs = read_csv(LOG_PATH)
    if days.empty or logs.empty:
        print("進捗確認に必要なCSVが足りません。")
        print()
        return

    latest = logs.drop_duplicates(subset=["place", "race_date_id"], keep="last")
    ok = latest[latest["status"] == "ok"]
    error = latest[latest["status"] == "error"]
    target_keys = set(zip(days["place"].astype(str), days["race_date_id"].astype(str)))
    ok_keys = set(zip(ok["place"].astype(str), ok["race_date_id"].astype(str)))
    done = len(target_keys & ok_keys)
    remaining = len(target_keys) - done
    print(f"対象開催: {len(target_keys):,}")
    print(f"成功開催: {done:,}")
    print(f"残り開催: {remaining:,}")
    print(f"最新error開催: {len(error):,}")
    print()


def show_feature_quality(features: pd.DataFrame) -> None:
    if features.empty:
        return
    print("=== features品質 ===")
    important_cols = [
        "avg_score", "max_score", "min_score", "std_score", "score_gap", "avg_age", "racer_count",
        "avg_win_rate", "max_win_rate", "avg_place_rate", "max_place_rate", "payout_3rentan", "payout_popularity",
        "is_over_30", "is_over_50", "is_over_100",
    ]
    existing = [c for c in important_cols if c in features.columns]
    nulls = features[existing].isna().sum().sort_values(ascending=False)
    print("欠損数:")
    print(nulls.to_string())
    print()

    if "payout_3rentan" in features.columns:
        payout = pd.to_numeric(features["payout_3rentan"], errors="coerce")
        print("配当分布:")
        print(payout.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
        print(f"3万円以上 : {(payout >= 30_000).sum():,}")
        print(f"5万円以上 : {(payout >= 50_000).sum():,}")
        print(f"10万円以上: {(payout >= 100_000).sum():,}")
        print()
        top_cols = [c for c in ["race_id", "combination", "payout_3rentan", "payout_popularity"] if c in features.columns]
        print("高配当トップ10:")
        print(features.assign(_payout=payout).sort_values("_payout", ascending=False)[top_cols].head(10).to_string(index=False))
        print()


def show_entries_quality(entries: pd.DataFrame) -> None:
    if entries.empty:
        return
    print("=== entries品質 ===")
    if "race_id" in entries.columns:
        racer_counts = entries.groupby("race_id").size()
        print("出走人数分布:")
        print(racer_counts.value_counts().sort_index().to_string())
        odd_counts = racer_counts[(racer_counts < 5) | (racer_counts > 9)]
        print(f"出走人数が5未満または10以上のrace_id: {len(odd_counts):,}")
    key_cols = [c for c in ["race_id", "car_no"] if c in entries.columns]
    if len(key_cols) == 2:
        print(f"race_id+car_no重複: {entries.duplicated(subset=key_cols).sum():,}")
    print()


def show_daily_files() -> None:
    print("=== 日別featuresファイル ===")
    if not DAY_ROOT.exists():
        print("日別フォルダがありません。")
        print()
        return
    files = list(DAY_ROOT.glob("*/*/features.csv"))
    empty = []
    for path in files:
        try:
            df = pd.read_csv(path)
            if df.empty:
                empty.append(path)
        except Exception:
            empty.append(path)
    print(f"features.csvファイル数: {len(files):,}")
    print(f"空または読めないfeatures.csv: {len(empty):,}")
    for path in empty[:10]:
        print(f"  {path}")
    print()


def main() -> None:
    features, entries, payouts = show_basic()
    show_fetch_progress()
    show_feature_quality(features)
    show_entries_quality(entries)
    show_daily_files()
    print("確認完了")


if __name__ == "__main__":
    main()
