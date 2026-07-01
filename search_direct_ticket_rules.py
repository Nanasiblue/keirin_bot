from __future__ import annotations

import pandas as pd

IN = "data/direct_ticket_predictions_full_valid.csv"
OUT_RULES = "data/direct_ticket_rule_search.csv"
OUT_MONTHLY = "data/direct_ticket_rule_search_monthly.csv"


def top_per_race(df, metric, n):
    return (
        df.sort_values(["race_id", metric], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(n)
    )


def summarize(sel, rule):
    races = sel["race_id"].nunique()
    tickets = len(sel)
    bet = tickets * 100
    ret = int(sel["return_yen"].sum()) if tickets else 0
    hit = int(sel["is_hit"].sum()) if tickets else 0
    max_hit = int(sel["return_yen"].max()) if tickets else 0

    return {
        "rule": rule,
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
        "avg_score": sel["direct_ticket_score"].mean() if tickets else 0,
        "min_score": sel["direct_ticket_score"].min() if tickets else 0,
        "avg_odds": sel["odds"].mean() if tickets else 0,
        "median_odds": sel["odds"].median() if tickets else 0,
        "avg_expected": sel["direct_expected_return"].mean() if tickets else 0,
        "max_hit_payout": max_hit,
        "max_hit_share": max_hit / ret if ret else 0,
    }


def monthly(sel, rule):
    rows = []
    for month, g in sel.groupby("month"):
        row = summarize(g, rule)
        row["month"] = month
        rows.append(row)
    return rows


def main():
    print("loading...")
    df = pd.read_csv(IN, dtype={"race_id": str, "combination": str})

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["month"] = df["race_date"].dt.to_period("M").astype(str)

    for col in ["odds", "direct_ticket_score", "direct_expected_return", "return_yen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)

    odds_ranges = [
        ("odds_30_500", 30, 500),
        ("odds_50_500", 50, 500),
        ("odds_50_1000", 50, 1000),
        ("odds_100_1000", 100, 1000),
        ("odds_30plus", 30, 999999),
        ("odds_50plus", 50, 999999),
        ("odds_100plus", 100, 999999),
    ]

    score_thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
    expected_thresholds = [0, 500, 1000, 1500, 2000, 3000, 5000]
    top_ns = [1, 2, 3, 5, 10]
    metrics = ["direct_ticket_score", "direct_expected_return"]

    rows = []
    month_rows = []

    print("searching rules...")
    for odds_name, lo, hi in odds_ranges:
        odds_base = df[(df["odds"] >= lo) & (df["odds"] <= hi)].copy()
        if odds_base.empty:
            continue

        for score_th in score_thresholds:
            score_base = odds_base[odds_base["direct_ticket_score"] >= score_th].copy()
            if score_base.empty:
                continue

            for exp_th in expected_thresholds:
                base = score_base[score_base["direct_expected_return"] >= exp_th].copy()
                if base.empty:
                    continue

                for metric in metrics:
                    for n in top_ns:
                        sel = top_per_race(base, metric, n)
                        rule = f"{odds_name}_score>={score_th}_expected>={exp_th}_top{n}_by_{metric}"
                        rows.append(summarize(sel, rule))
                        month_rows.extend(monthly(sel, rule))

    result = pd.DataFrame(rows)
    monthly_result = pd.DataFrame(month_rows)

    if not monthly_result.empty:
        month_roi = monthly_result.pivot_table(
            index="rule",
            values="roi",
            aggfunc=["min", "mean", "count"],
        )
        month_roi.columns = ["month_min_roi", "month_mean_roi", "month_count"]

        result = result.merge(month_roi.reset_index(), on="rule", how="left")

        monthly_positive = (
            monthly_result.assign(is_plus=monthly_result["roi"] >= 1.0)
            .groupby("rule")["is_plus"]
            .sum()
        )

        result = result.merge(
            monthly_positive.rename("plus_months").reset_index(),
            on="rule",
            how="left",
        )

    result = result.sort_values(
        ["roi_without_max_hit", "month_min_roi", "roi", "hit_count"],
        ascending=[False, False, False, False],
    )

    result.to_csv(OUT_RULES, index=False, encoding="utf-8-sig")
    monthly_result.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT_RULES}")
    print(f"saved: {OUT_MONTHLY}")

    print(result.head(80).to_string(
        index=False,
        formatters={
            "roi": lambda x: f"{x:.2%}",
            "roi_without_max_hit": lambda x: f"{x:.2%}",
            "month_min_roi": lambda x: f"{x:.2%}",
            "month_mean_roi": lambda x: f"{x:.2%}",
            "hit_rate_per_ticket": lambda x: f"{x:.2%}",
            "hit_rate_per_race": lambda x: f"{x:.2%}",
            "max_hit_share": lambda x: f"{x:.2%}",
            "avg_tickets_per_race": lambda x: f"{x:.1f}",
            "avg_score": lambda x: f"{x:.4f}",
            "min_score": lambda x: f"{x:.4f}",
            "avg_odds": lambda x: f"{x:.1f}",
            "median_odds": lambda x: f"{x:.1f}",
            "avg_expected": lambda x: f"{x:.1f}",
        },
    ))


if __name__ == "__main__":
    main()
