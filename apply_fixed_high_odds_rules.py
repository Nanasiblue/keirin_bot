from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TICKETS = "data/direct_ticket_predictions_full_valid.csv"
DEFAULT_RACE_SCORE = "data/predictions_valid_2026_is_over_50.csv"
DEFAULT_OUTPUT = "data/fixed_rule_high_odds_v1.csv"
DEFAULT_MONTHLY = "data/fixed_rule_high_odds_v1_monthly.csv"
DEFAULT_DAILY = "data/fixed_rule_high_odds_v1_daily.csv"
DEFAULT_EQUITY = "data/fixed_rule_high_odds_v1_equity.csv"


RULES = [
    {
        "rule": "high_odds_v1_top2",
        "odds_min": 100,
        "odds_max": 500,
        "race_score_min": 0.45,
        "ticket_score_min": 0.40,
        "expected_min": 3000,
        "top_n": 2,
        "sort_col": "direct_ticket_score",
    },
    {
        "rule": "high_odds_v1_top5",
        "odds_min": 100,
        "odds_max": 500,
        "race_score_min": 0.45,
        "ticket_score_min": 0.40,
        "expected_min": 3000,
        "top_n": 5,
        "sort_col": "expected_return",
    },
]


LEAK_COLUMNS = {
    "is_over_30",
    "is_over_50",
    "is_over_100",
    "payout_3rentan",
    "payout_popularity",
    "payout",
    "popularity",
    "combination",
    "target",
    "label",
    "actual",
    "hit",
    "is_hit",
    "return",
}


def pct(value: float) -> str:
    return f"{value:.2%}"


def detect_race_score_col(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"--race-score-col が見つかりません: {explicit}")
        if explicit in LEAK_COLUMNS or explicit.startswith("is_over_"):
            raise SystemExit(f"危険: {explicit} は正解列っぽいので使いません")
        return explicit

    preferred = ["ai_score", "race_score", "race_ai_score", "pred_score", "score", "prob", "proba"]
    for col in preferred:
        if col in df.columns and col not in LEAK_COLUMNS and not col.startswith("is_over_"):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col

    candidates = []
    for col in df.columns:
        low = col.lower()
        if col == "race_id" or col in LEAK_COLUMNS or col.startswith("is_over_"):
            continue
        if any(x in low for x in ["payout", "popularity", "combination"]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)

    print("race score列を自動判定できませんでした。候補:")
    for col in candidates:
        s = df[col]
        print(f"  {col}: min={s.min()} max={s.max()} nunique={s.nunique()}")
    raise SystemExit("例: python apply_fixed_high_odds_rules.py --race-score-col ai_score")


def summarize(df: pd.DataFrame, name: str) -> dict:
    races = df["race_id"].nunique()
    tickets = len(df)
    bet = tickets * 100
    ret = int(df["return_yen"].sum()) if tickets else 0
    hit = int(df["is_hit"].sum()) if tickets else 0
    max_hit = int(df["return_yen"].max()) if tickets else 0
    profit = ret - bet

    return {
        "rule": name,
        "race_count": races,
        "tickets": tickets,
        "avg_tickets_per_race": tickets / races if races else 0,
        "bet": bet,
        "return": ret,
        "profit": profit,
        "roi": ret / bet if bet else 0,
        "roi_without_max_hit": (ret - max_hit) / bet if bet else 0,
        "hit_count": hit,
        "hit_rate_per_ticket": hit / tickets if tickets else 0,
        "hit_rate_per_race": hit / races if races else 0,
        "avg_race_score": df["race_score"].mean() if tickets else 0,
        "min_race_score": df["race_score"].min() if tickets else 0,
        "avg_score": df["direct_ticket_score"].mean() if tickets else 0,
        "min_score": df["direct_ticket_score"].min() if tickets else 0,
        "avg_odds": df["odds"].mean() if tickets else 0,
        "median_odds": df["odds"].median() if tickets else 0,
        "avg_expected": df["expected_return"].mean() if tickets else 0,
        "max_hit_payout": max_hit,
        "max_hit_share": max_hit / ret if ret else 0,
    }


def max_losing_streak(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    race_returns = (
        df.groupby(["race_date", "race_id"], as_index=False)
        .agg(bet=("combination", lambda s: len(s) * 100), ret=("return_yen", "sum"))
        .sort_values(["race_date", "race_id"])
    )

    streak = 0
    worst = 0
    for _, row in race_returns.iterrows():
        if row["ret"] - row["bet"] < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def apply_rule(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    base = df[
        (df["odds"] >= rule["odds_min"])
        & (df["odds"] <= rule["odds_max"])
        & (df["race_score"] >= rule["race_score_min"])
        & (df["direct_ticket_score"] >= rule["ticket_score_min"])
        & (df["expected_return"] >= rule["expected_min"])
    ].copy()

    return (
        base.sort_values(["race_id", rule["sort_col"]], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(rule["top_n"])
        .copy()
    )


def add_display_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for col in ["roi", "roi_without_max_hit", "hit_rate_per_ticket", "hit_rate_per_race", "max_hit_share"]:
        if col in out.columns:
            out[col] = out[col].map(pct)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", default=DEFAULT_TICKETS)
    parser.add_argument("--race-score", default=DEFAULT_RACE_SCORE)
    parser.add_argument("--race-score-col", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--monthly-output", default=DEFAULT_MONTHLY)
    parser.add_argument("--daily-output", default=DEFAULT_DAILY)
    parser.add_argument("--equity-output", default=DEFAULT_EQUITY)
    args = parser.parse_args()

    print("loading ticket predictions...")
    df = pd.read_csv(args.tickets, dtype={"race_id": str, "combination": str})
    print(f"tickets: {len(df):,} / races: {df['race_id'].nunique():,}")

    print("loading race score...")
    race = pd.read_csv(args.race_score, dtype={"race_id": str})
    race_col = detect_race_score_col(race, args.race_score_col)
    race = race[["race_id", race_col]].rename(columns={race_col: "race_score"})
    print(f"race score column: {race_col}")

    df = df.merge(race, on="race_id", how="left")
    missing = int(df["race_score"].isna().sum())
    if missing:
        print(f"race_score missing: {missing:,}")
    df = df.dropna(subset=["race_score"]).copy()

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["month"] = df["race_date"].dt.to_period("M").astype(str)
    df["day"] = df["race_date"].dt.strftime("%Y-%m-%d")

    for col in ["odds", "direct_ticket_score", "return_yen", "race_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)
    df["expected_return"] = df["direct_ticket_score"] * df["odds"] * 100

    selected_list = []
    summary_rows = []
    monthly_rows = []
    daily_rows = []
    equity_rows = []

    for rule in RULES:
        picked = apply_rule(df, rule)
        picked.insert(0, "rule", rule["rule"])
        selected_list.append(picked)

        row = summarize(picked, rule["rule"])
        row["max_losing_streak_races"] = max_losing_streak(picked)
        summary_rows.append(row)

        for month, g in picked.groupby("month"):
            m = summarize(g, rule["rule"])
            m["month"] = month
            monthly_rows.append(m)

        for day, g in picked.groupby("day"):
            d = summarize(g, rule["rule"])
            d["day"] = day
            daily_rows.append(d)

        race_returns = (
            picked.groupby(["race_date", "race_id"], as_index=False)
            .agg(tickets=("combination", "size"), ret=("return_yen", "sum"), hits=("is_hit", "sum"))
            .sort_values(["race_date", "race_id"])
        )
        race_returns["rule"] = rule["rule"]
        race_returns["bet"] = race_returns["tickets"] * 100
        race_returns["profit"] = race_returns["ret"] - race_returns["bet"]
        race_returns["cum_profit"] = race_returns["profit"].cumsum()
        equity_rows.append(race_returns)

    selected = pd.concat(selected_list, ignore_index=True) if selected_list else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    daily = pd.DataFrame(daily_rows)
    equity = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")
    monthly.to_csv(args.monthly_output, index=False, encoding="utf-8-sig")
    daily.to_csv(args.daily_output, index=False, encoding="utf-8-sig")
    equity.to_csv(args.equity_output, index=False, encoding="utf-8-sig")

    print("")
    print("=== fixed high odds summary ===")
    print(
        add_display_columns(summary).to_string(
            index=False,
            formatters={
                "avg_tickets_per_race": lambda x: f"{x:.1f}",
                "avg_race_score": lambda x: f"{x:.4f}",
                "min_race_score": lambda x: f"{x:.4f}",
                "avg_score": lambda x: f"{x:.4f}",
                "min_score": lambda x: f"{x:.4f}",
                "avg_odds": lambda x: f"{x:.1f}",
                "median_odds": lambda x: f"{x:.1f}",
                "avg_expected": lambda x: f"{x:.1f}",
            },
        )
    )

    if not monthly.empty:
        print("")
        print("=== monthly ===")
        print(
            add_display_columns(monthly).to_string(
                index=False,
                formatters={
                    "avg_tickets_per_race": lambda x: f"{x:.1f}",
                    "avg_race_score": lambda x: f"{x:.4f}",
                    "min_race_score": lambda x: f"{x:.4f}",
                    "avg_score": lambda x: f"{x:.4f}",
                    "min_score": lambda x: f"{x:.4f}",
                    "avg_odds": lambda x: f"{x:.1f}",
                    "median_odds": lambda x: f"{x:.1f}",
                    "avg_expected": lambda x: f"{x:.1f}",
                },
            )
        )

    print("")
    print(f"saved: {args.output}")
    print(f"saved: {args.monthly_output}")
    print(f"saved: {args.daily_output}")
    print(f"saved: {args.equity_output}")


if __name__ == "__main__":
    main()
