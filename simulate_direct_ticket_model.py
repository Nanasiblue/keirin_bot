from __future__ import annotations

import pandas as pd


IN = "data/direct_ticket_predictions_valid.csv"
OUT = "data/direct_ticket_model_simulation.csv"


def top_per_race(df, metric, n):
    return (
        df.sort_values(["race_id", metric], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(n)
    )


def simulate(selected, rule):
    race_count = selected["race_id"].nunique()
    tickets = len(selected)
    bet = tickets * 100
    ret = int(selected["return_yen"].sum())
    hit = int(selected["is_hit"].sum())
    max_hit = int(selected["return_yen"].max()) if tickets else 0

    return {
        "rule": rule,
        "race_count": race_count,
        "tickets": tickets,
        "avg_tickets_per_race": tickets / race_count if race_count else 0,
        "bet": bet,
        "return": ret,
        "profit": ret - bet,
        "roi": ret / bet if bet else 0,
        "roi_without_max_hit": (ret - max_hit) / bet if bet else 0,
        "hit_count": hit,
        "hit_rate_per_ticket": hit / tickets if tickets else 0,
        "hit_rate_per_race": hit / race_count if race_count else 0,
        "avg_score": selected["direct_ticket_score"].mean() if tickets else 0,
        "min_score": selected["direct_ticket_score"].min() if tickets else 0,
        "avg_odds": selected["odds"].mean() if tickets else 0,
        "median_odds": selected["odds"].median() if tickets else 0,
        "avg_direct_expected_return": selected["direct_expected_return"].mean() if tickets else 0,
        "avg_hit_payout": selected.loc[selected["is_hit"].eq(1), "return_yen"].mean() if hit else 0,
        "max_hit_payout": max_hit,
        "max_hit_share": max_hit / ret if ret else 0,
    }


def main():
    print("loading...")
    df = pd.read_csv(IN, dtype={"race_id": str, "combination": str})

    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(0)
    df["estimated_payout"] = pd.to_numeric(df["estimated_payout"], errors="coerce").fillna(0)
    df["direct_ticket_score"] = pd.to_numeric(df["direct_ticket_score"], errors="coerce").fillna(0)
    df["direct_expected_return"] = pd.to_numeric(df["direct_expected_return"], errors="coerce").fillna(0)
    df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)
    df["hit_payout"] = pd.to_numeric(df["hit_payout"], errors="coerce").fillna(0)
    df["return_yen"] = df["is_hit"] * df["hit_payout"]

    print(f"rows: {len(df):,}")
    print(f"races: {df['race_id'].nunique():,}")
    print(f"hit rows: {int(df['is_hit'].sum()):,}")
    print(f"base hit rate: {df['is_hit'].mean():.4%}")

    rows = []

    odds_ranges = [
        ("odds_all", 0, 999999),
        ("odds_10plus", 10, 999999),
        ("odds_30plus", 30, 999999),
        ("odds_50plus", 50, 999999),
        ("odds_100plus", 100, 999999),
        ("odds_30_500", 30, 500),
        ("odds_50_500", 50, 500),
        ("odds_50_1000", 50, 1000),
        ("odds_100_1000", 100, 1000),
    ]

    top_ns = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
    score_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    expected_thresholds = [50, 80, 100, 150, 200, 300, 500, 800, 1000]

    for odds_name, lo, hi in odds_ranges:
        base = df[(df["odds"] >= lo) & (df["odds"] <= hi)].copy()
        if base.empty:
            continue

        for metric in ["direct_ticket_score", "direct_expected_return"]:
            for n in top_ns:
                sel = top_per_race(base, metric, n)
                rows.append(simulate(sel, f"{odds_name}_top{n}_by_{metric}"))

        for th in score_thresholds:
            sel_base = base[base["direct_ticket_score"] >= th].copy()
            if not sel_base.empty:
                rows.append(simulate(sel_base, f"{odds_name}_score>={th}"))
                for n in [1, 2, 3, 5, 10]:
                    sel = top_per_race(sel_base, "direct_ticket_score", n)
                    rows.append(simulate(sel, f"{odds_name}_score>={th}_top{n}"))

        for th in expected_thresholds:
            sel_base = base[base["direct_expected_return"] >= th].copy()
            if not sel_base.empty:
                rows.append(simulate(sel_base, f"{odds_name}_expected>={th}"))
                for n in [1, 2, 3, 5, 10]:
                    sel = top_per_race(sel_base, "direct_expected_return", n)
                    rows.append(simulate(sel, f"{odds_name}_expected>={th}_top{n}"))

    out = pd.DataFrame(rows).sort_values(
        ["roi_without_max_hit", "roi", "hit_count"],
        ascending=[False, False, False],
    )

    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT}")

    print(out.head(80).to_string(
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
            "avg_direct_expected_return": lambda x: f"{x:.1f}",
            "avg_hit_payout": lambda x: f"{x:.0f}",
            "max_hit_payout": lambda x: f"{x:.0f}",
        },
    ))


if __name__ == "__main__":
    main()
