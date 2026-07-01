from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

def parse_range(text):
    a, b = text.replace(" ", "").split("-")
    return int(a), int(b)

def select_races(df, rule):
    if rule.startswith("score>="):
        return df[df["ai_score"] >= float(rule.split(">=")[1])].copy()
    if rule.startswith("top_") and rule.endswith("pct"):
        pct = float(rule.replace("top_", "").replace("pct", "")) / 100
        return df.head(max(1, int(len(df) * pct))).copy()
    if rule.startswith("top_"):
        return df.head(int(rule.replace("top_", ""))).copy()
    return df.copy()

def simulate(selected, race_rule, pop_range, bet_unit):
    lo, hi = pop_range
    points = hi - lo + 1
    races = len(selected)
    bet = races * points * bet_unit
    hit = selected[
        (selected["payout_popularity"] >= lo)
        & (selected["payout_popularity"] <= hi)
    ]
    ret = hit["payout_3rentan"].sum() * (bet_unit / 100)

    return {
        "race_rule": race_rule,
        "popularity_range": f"{lo}-{hi}",
        "race_count": races,
        "points_per_race": points,
        "total_bet": int(bet),
        "hit_count": len(hit),
        "hit_rate": len(hit) / races if races else 0,
        "total_return": int(ret),
        "roi": ret / bet if bet else 0,
        "profit": int(ret - bet),
        "avg_hit_payout": hit["payout_3rentan"].mean() if len(hit) else 0,
        "max_hit_payout": hit["payout_3rentan"].max() if len(hit) else 0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/predictions_valid_2026_is_over_50.csv")
    parser.add_argument("--output", default="data/betting_popularity_simulation_is_over_50.csv")
    parser.add_argument("--bet-unit", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["ai_score"] = pd.to_numeric(df["ai_score"], errors="coerce")
    df["payout_3rentan"] = pd.to_numeric(df["payout_3rentan"], errors="coerce")
    df["payout_popularity"] = pd.to_numeric(df["payout_popularity"], errors="coerce")
    df = df.dropna(subset=["ai_score", "payout_3rentan", "payout_popularity"]).copy()
    df = df.sort_values("ai_score", ascending=False)
    df["payout_popularity"] = df["payout_popularity"].astype(int)

    race_rules = [
        "top_20", "top_30", "top_50", "top_100", "top_300",
        "top_3pct", "top_5pct",
        "score>=0.60", "score>=0.55", "score>=0.50",
    ]
    pop_ranges = [
        "1-50", "1-100", "20-100", "30-150", "30-200",
        "50-200", "50-300", "100-300", "150-400",
    ]

    rows = []
    for race_rule in race_rules:
        selected = select_races(df, race_rule)
        for pop_range in pop_ranges:
            rows.append(simulate(selected, race_rule, parse_range(pop_range), args.bet_unit))

    result = pd.DataFrame(rows)
    result = result.sort_values(["roi", "hit_count"], ascending=[False, False])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"saved: {args.output}")
    print(result.head(40).to_string(index=False, formatters={
        "hit_rate": "{:.2%}".format,
        "roi": "{:.2%}".format,
        "avg_hit_payout": "{:.0f}".format,
        "max_hit_payout": "{:.0f}".format,
    }))

if __name__ == "__main__":
    main()
