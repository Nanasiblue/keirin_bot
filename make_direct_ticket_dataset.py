from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ODDS = Path("data/odds_3rentan_all.csv")
ENTRIES = Path("data/entries_all_kdreams.csv")
PAYOUTS = Path("data/payouts_all_kdreams.csv")
FEATURES = Path("data/features_rich.csv")
OUT = Path("data/direct_ticket_dataset.csv")


def stable_random01(s, seed):
    h = pd.util.hash_pandas_object(s.astype(str) + f"_{seed}", index=False).astype("uint64")
    return (h % 1_000_000) / 1_000_000.0


def split_combination(df):
    cars = df["combination"].astype(str).str.split("-", expand=True)
    df["first_car"] = pd.to_numeric(cars[0], errors="coerce").astype("Int64")
    df["second_car"] = pd.to_numeric(cars[1], errors="coerce").astype("Int64")
    df["third_car"] = pd.to_numeric(cars[2], errors="coerce").astype("Int64")
    return df


def race_date_from_id(race_id):
    start = pd.to_datetime(race_id.str[2:10], format="%Y%m%d", errors="coerce")
    day_no = pd.to_numeric(race_id.str[10:12], errors="coerce").fillna(1).astype(int)
    return start + pd.to_timedelta(day_no - 1, unit="D")


def load_entries():
    e = pd.read_csv(ENTRIES, dtype={"race_id": str})
    e["car_no"] = pd.to_numeric(e["car_no"], errors="coerce").astype("Int64")
    for c in ["score", "age", "win_rate", "place_rate"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    return e[["race_id", "car_no", "score", "style", "age", "win_rate", "place_rate"]]


def add_car_features(df, entries):
    for pos, car_col in [("first", "first_car"), ("second", "second_car"), ("third", "third_car")]:
        tmp = entries.rename(columns={
            "car_no": car_col,
            "score": f"{pos}_score",
            "style": f"{pos}_style",
            "age": f"{pos}_age",
            "win_rate": f"{pos}_win_rate",
            "place_rate": f"{pos}_place_rate",
        })
        df = df.merge(tmp, on=["race_id", car_col], how="left")
    return df


def add_ticket_features(df):
    df["score_1_2_gap"] = df["first_score"] - df["second_score"]
    df["score_1_3_gap"] = df["first_score"] - df["third_score"]
    df["score_2_3_gap"] = df["second_score"] - df["third_score"]
    df["score_ticket_avg"] = df[["first_score", "second_score", "third_score"]].mean(axis=1)
    df["win_rate_ticket_avg"] = df[["first_win_rate", "second_win_rate", "third_win_rate"]].mean(axis=1)
    df["place_rate_ticket_avg"] = df[["first_place_rate", "second_place_rate", "third_place_rate"]].mean(axis=1)
    return df


def load_hit_map():
    p = pd.read_csv(PAYOUTS, dtype={"race_id": str, "combination": str})
    p = p[p["bet_type"].astype(str).eq("3連単")].copy()
    p = p[["race_id", "combination", "payout", "popularity"]].rename(columns={
        "combination": "hit_combination",
        "payout": "hit_payout",
        "popularity": "hit_popularity",
    })
    return p


def load_race_features():
    f = pd.read_csv(FEATURES, dtype={"race_id": str})
    keep = [
        "race_id", "avg_score", "max_score", "min_score", "std_score", "score_gap",
        "avg_age", "racer_count", "avg_win_rate", "max_win_rate",
        "avg_place_rate", "max_place_rate", "nige_count", "oikomi_count",
        "ryo_count", "sashi_count", "makuri_count", "front_runner_pressure",
        "place", "race_no", "weather", "wind_speed", "is_rain", "is_strong_wind",
        "payout_3rentan", "payout_popularity", "is_over_30", "is_over_50", "is_over_100",
    ]
    keep = [c for c in keep if c in f.columns]
    return f[keep]


def select_rows(chunk, hit, neg_frac, seed):
    chunk = chunk.merge(hit, on="race_id", how="left")
    chunk["is_hit"] = (chunk["combination"] == chunk["hit_combination"]).astype(int)

    chunk["odds"] = pd.to_numeric(chunk["odds"], errors="coerce")
    chunk["estimated_payout"] = pd.to_numeric(chunk["estimated_payout"], errors="coerce")

    key = chunk["race_id"].astype(str) + "_" + chunk["combination"].astype(str)
    r = stable_random01(key, seed)

    keep_negative = (
        (r < neg_frac)
        | ((chunk["odds"] >= 30) & (chunk["odds"] <= 500) & (r < neg_frac * 2.0))
        | ((chunk["odds"] >= 100) & (chunk["odds"] <= 1000) & (r < neg_frac * 2.5))
    )

    return chunk[chunk["is_hit"].eq(1) | keep_negative].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negative-frac", type=float, default=0.12)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.replace:
        raise SystemExit(f"{OUT} already exists. 上書きするなら --replace を付けてください。")
    if OUT.exists() and args.replace:
        OUT.unlink()

    print("loading entries...")
    entries = load_entries()

    print("loading hit payouts...")
    hit = load_hit_map()

    print("loading race features...")
    race_features = load_race_features()

    total_in = 0
    total_out = 0
    total_hit = 0
    write_header = True

    for i, chunk in enumerate(pd.read_csv(ODDS, dtype={"race_id": str, "combination": str}, chunksize=args.chunksize), start=1):
        total_in += len(chunk)

        sampled = select_rows(chunk, hit, args.negative_frac, args.seed)
        if sampled.empty:
            continue

        sampled = split_combination(sampled)
        sampled = add_car_features(sampled, entries)
        sampled = add_ticket_features(sampled)
        sampled = sampled.merge(race_features, on="race_id", how="left")

        sampled["race_date"] = race_date_from_id(sampled["race_id"])
        sampled["year"] = sampled["race_date"].dt.year
        sampled["month"] = sampled["race_date"].dt.month

        total_out += len(sampled)
        total_hit += int(sampled["is_hit"].sum())

        sampled.to_csv(
            OUT,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
            encoding="utf-8-sig" if write_header else "utf-8",
        )
        write_header = False

        print(f"chunk {i}: scanned={total_in:,} output={total_out:,} hits={total_hit:,} positive_rate={total_hit / total_out:.4%}")

        if args.limit_chunks and i >= args.limit_chunks:
            break

    print("done")
    print(f"saved: {OUT}")
    print(f"scanned rows: {total_in:,}")
    print(f"dataset rows: {total_out:,}")
    print(f"hit rows: {total_hit:,}")
    if total_out:
        print(f"positive rate: {total_hit / total_out:.4%}")


if __name__ == "__main__":
    main()
