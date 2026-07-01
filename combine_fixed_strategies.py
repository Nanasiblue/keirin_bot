from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MIDDLE = "data/fixed_rule_direct_v1_middle_odds.csv"
DEFAULT_HIGH = "data/fixed_rule_high_odds_v1.csv"
DEFAULT_OUTPUT = "data/combined_strategy_middle_plus_high_top5.csv"
DEFAULT_SUMMARY = "data/combined_strategy_summary.csv"
DEFAULT_MONTHLY = "data/combined_strategy_monthly.csv"
DEFAULT_DAILY = "data/combined_strategy_daily.csv"
DEFAULT_EQUITY = "data/combined_strategy_equity.csv"

STRATEGY_NAME = "middle_plus_high_top5"
MIDDLE_RULE = "direct_v1_middle_odds"
HIGH_RULE = "high_odds_v1_top5"


def pct(value):
    return f"{value:.2%}"


def normalize(df, rule_name=None):
    df = df.copy()
    if "rule" not in df.columns:
        df.insert(0, "rule", rule_name or "unknown")
    elif rule_name is not None:
        df["rule"] = rule_name

    df["race_id"] = df["race_id"].astype(str)
    df["combination"] = df["combination"].astype(str)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["month"] = df["race_date"].dt.to_period("M").astype(str)
    df["day"] = df["race_date"].dt.strftime("%Y-%m-%d")

    for col in ["odds", "return_yen", "direct_ticket_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else 0

    df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)
    return df


def summarize(df, name):
    races = df["race_id"].nunique()
    tickets = len(df)
    bet = tickets * 100
    ret = int(df["return_yen"].sum()) if tickets else 0
    hit = int(df["is_hit"].sum()) if tickets else 0
    max_hit = int(df["return_yen"].max()) if tickets else 0

    return {
        "strategy": name,
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
        "avg_odds": df["odds"].mean() if tickets else 0,
        "median_odds": df["odds"].median() if tickets else 0,
        "avg_score": df["direct_ticket_score"].mean() if tickets else 0,
        "max_hit_payout": max_hit,
        "max_hit_share": max_hit / ret if ret else 0,
    }


def make_equity(df):
    eq = (
        df.groupby(["race_date", "race_id"], as_index=False)
        .agg(tickets=("combination", "size"), return_yen=("return_yen", "sum"), hits=("is_hit", "sum"))
        .sort_values(["race_date", "race_id"])
    )
    eq["bet"] = eq["tickets"] * 100
    eq["profit"] = eq["return_yen"] - eq["bet"]
    eq["cum_profit"] = eq["profit"].cumsum()
    eq["running_peak"] = eq["cum_profit"].cummax()
    eq["drawdown"] = eq["cum_profit"] - eq["running_peak"]
    return eq


def risk_stats(eq):
    streak = 0
    worst = 0
    for _, row in eq.iterrows():
        if row["profit"] < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0

    daily = eq.copy()
    daily["day"] = daily["race_date"].dt.strftime("%Y-%m-%d")
    day_profit = daily.groupby("day")["profit"].sum()

    return {
        "max_losing_streak_races": int(worst),
        "max_drawdown": int(eq["drawdown"].min()) if len(eq) else 0,
        "worst_day_profit": int(day_profit.min()) if len(day_profit) else 0,
        "best_day_profit": int(day_profit.max()) if len(day_profit) else 0,
    }


def display(df):
    out = df.copy()
    for col in ["roi", "roi_without_max_hit", "hit_rate_per_ticket", "hit_rate_per_race", "max_hit_share"]:
        if col in out.columns:
            out[col] = out[col].map(pct)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", default=DEFAULT_MIDDLE)
    parser.add_argument("--high", default=DEFAULT_HIGH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--monthly-output", default=DEFAULT_MONTHLY)
    parser.add_argument("--daily-output", default=DEFAULT_DAILY)
    parser.add_argument("--equity-output", default=DEFAULT_EQUITY)
    args = parser.parse_args()

    middle = normalize(pd.read_csv(args.middle, dtype={"race_id": str, "combination": str}), MIDDLE_RULE)

    high = normalize(pd.read_csv(args.high, dtype={"race_id": str, "combination": str}))
    high = high[high["rule"] == HIGH_RULE].copy()

    print(f"middle tickets: {len(middle):,} / races: {middle['race_id'].nunique():,}")
    print(f"high tickets  : {len(high):,} / races: {high['race_id'].nunique():,}")

    raw = pd.concat([middle, high], ignore_index=True, sort=False)
    before = len(raw)

    raw["source_rules"] = raw.groupby(["race_id", "combination"])["rule"].transform(
        lambda s: "+".join(sorted(set(s.astype(str))))
    )

    combined = (
        raw.sort_values(["race_date", "race_id", "combination", "return_yen"], ascending=[True, True, True, False])
        .drop_duplicates(["race_id", "combination"], keep="first")
        .copy()
    )
    combined["strategy"] = STRATEGY_NAME
    duplicates = before - len(combined)

    eq = make_equity(combined)
    risk = risk_stats(eq)

    summary_rows = []
    for name, group in [(MIDDLE_RULE, middle), (HIGH_RULE, high), (STRATEGY_NAME, combined)]:
        row = summarize(group, name)
        if name == STRATEGY_NAME:
            row.update(risk)
            row["deduplicated_tickets"] = duplicates
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    monthly_rows = []
    for name, group in [(MIDDLE_RULE, middle), (HIGH_RULE, high), (STRATEGY_NAME, combined)]:
        for month, g in group.groupby("month"):
            row = summarize(g, name)
            row["month"] = month
            monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)

    daily_rows = []
    for day, g in combined.groupby("day"):
        row = summarize(g, STRATEGY_NAME)
        row["day"] = day
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    monthly.to_csv(args.monthly_output, index=False, encoding="utf-8-sig")
    daily.to_csv(args.daily_output, index=False, encoding="utf-8-sig")
    eq.to_csv(args.equity_output, index=False, encoding="utf-8-sig")

    print("")
    print("=== combined summary ===")
    print(display(summary).to_string(index=False))

    print("")
    print("=== combined monthly ===")
    print(display(monthly[monthly["strategy"] == STRATEGY_NAME]).to_string(index=False))

    print("")
    print(f"deduplicated tickets: {duplicates}")
    print(f"saved: {args.output}")
    print(f"saved: {args.summary_output}")
    print(f"saved: {args.monthly_output}")
    print(f"saved: {args.daily_output}")
    print(f"saved: {args.equity_output}")


if __name__ == "__main__":
    main()
