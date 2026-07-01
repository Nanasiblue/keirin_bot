from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

DROP_COLS = {
    "race_id", "race_date_id", "combination", "payout_3rentan", "payout_popularity",
    "is_over_30", "is_over_50", "is_over_100",
    "is_popular_50plus", "is_popular_100plus", "is_popular_150plus", "parse_error",
}

def add_race_date(df):
    df = df.copy()
    race_id = df["race_id"].astype(str).str.zfill(16)
    start_date = pd.to_datetime(race_id.str[2:10], format="%Y%m%d", errors="coerce")
    day_no = pd.to_numeric(race_id.str[10:12], errors="coerce").fillna(1).astype(int)
    df["race_date"] = start_date + pd.to_timedelta(day_no - 1, unit="D")
    return df

def build_x(df, columns):
    x = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()
    x = x.drop(columns=["race_date", "race_start_date"], errors="ignore")
    cats = [c for c in ["place", "weather"] if c in x.columns]
    x = pd.get_dummies(x, columns=cats, dummy_na=True)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)
    return x.reindex(columns=columns, fill_value=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/random_forest_tuned_is_over_50.joblib")
    parser.add_argument("--input", default="data/features_rich.csv")
    parser.add_argument("--output", default="data/predictions_is_over_50.csv")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    columns = bundle["columns"]

    df = pd.read_csv(args.input, dtype={"race_id": str, "race_date_id": str})
    df = add_race_date(df)

    if args.start:
        df = df[df["race_date"] >= pd.Timestamp(args.start)].copy()
    if args.end:
        df = df[df["race_date"] <= pd.Timestamp(args.end)].copy()

    df["ai_score"] = model.predict_proba(build_x(df, columns))[:, 1]
    out = df.sort_values("ai_score", ascending=False)

    keep = [
        "race_id", "race_date", "place", "race_no", "grade", "weather", "wind_speed",
        "ai_score", "combination", "payout_3rentan", "payout_popularity",
        "is_over_30", "is_over_50", "is_over_100",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"saved: {args.output}")
    print(out.head(args.top).to_string(index=False))

if __name__ == "__main__":
    main()
