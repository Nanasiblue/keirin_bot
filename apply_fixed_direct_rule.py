from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "data/direct_ticket_predictions_full_valid.csv"
DEFAULT_OUTPUT = "data/fixed_rule_direct_v1_middle_odds.csv"
DEFAULT_MONTHLY = "data/fixed_rule_direct_v1_middle_odds_monthly.csv"


def summarize(df: pd.DataFrame, name: str) -> dict:
    races = df["race_id"].nunique()
    tickets = len(df)
    bet = tickets * 100
    ret = int(df["return_yen"].sum()) if tickets else 0
    hit = int(df["is_hit"].sum()) if tickets else 0
    max_hit = int(df["return_yen"].max()) if tickets else 0

    return {
        "rule": name,
        "race_count": races,
        "tickets": tickets,
        "avg_tickets_per_race": tickets / races if races else 0,
        "bet": bet,
        "return": ret,
        "profit": ret - bet,
        "roi": ret / bet if bet else 0,
        "roi_without_max_hit": (ret - max_hit) / bet if bet else 0,
        "hit_count": hit,
        "hit_rate_per_ticket": hit / tickets if tickets else 0,
        "hit_rate_per_race": hit / races if races else 0,
        "avg_score": df["direct_ticket_score"].mean() if tickets else 0,
        "min_score": df["direct_ticket_score"].min() if tickets else 0,
        "avg_odds": df["odds"].mean() if tickets else 0,
        "median_odds": df["odds"].median() if tickets else 0,
        "max_hit_payout": max_hit,
        "max_hit_share": max_hit / ret if ret else 0,
    }


def apply_rule(df: pd.DataFrame) -> pd.DataFrame:
    base = df[
        (df["odds"] >= 30)
        & (df["odds"] <= 500)
        & (df["direct_ticket_score"] >= 0.9)
    ].copy()

    selected = (
        base.sort_values(["race_id", "direct_ticket_score"], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(2)
        .copy()
    )

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--monthly-output", default=DEFAULT_MONTHLY)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"race_id": str, "combination": str})

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["month"] = df["race_date"].dt.to_period("M").astype(str)

    for col in ["odds", "direct_ticket_score", "return_yen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)

    selected = apply_rule(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False, encoding="utf-8-sig")

    summary = summarize(selected, "direct_v1_middle_odds")

    monthly_rows = []
    for month, g in selected.groupby("month"):
        row = summarize(g, "direct_v1_middle_odds")
        row["month"] = month
        monthly_rows.append(row)

    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(args.monthly_output, index=False, encoding="utf-8-sig")

    print("=== fixed rule summary ===")
    for key, value in summary.items():
        if key in ["roi", "roi_without_max_hit", "hit_rate_per_ticket", "hit_rate_per_race", "max_hit_share"]:
            print(f"{key}: {value:.2%}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"saved: {out}")
    print(f"saved: {args.monthly_output}")

    if not monthly.empty:
        print("")
        print("=== monthly ===")
        print(monthly.to_string(
            index=False,
            formatters={
                "roi": lambda x: f"{x:.2%}",
                "roi_without_max_hit": lambda x: f"{x:.2%}",
                "hit_rate_per_ticket": lambda x: f"{x:.2%}",
                "hit_rate_per_race": lambda x: f"{x:.2%}",
                "max_hit_share": lambda x: f"{x:.2%}",
                "avg_tickets_per_race": lambda x: f"{x:.1f}",
                "avg_score": lambda x: f"{x:.4f}",
                "min_score": lambda x: f"{x:.4f}",
                "avg_odds": lambda x: f"{x:.1f}",
                "median_odds": lambda x: f"{x:.1f}",
            },
        ))


if __name__ == "__main__":
    main()
