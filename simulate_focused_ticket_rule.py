from __future__ import annotations

import pandas as pd


IN = "data/ticket_scores_valid_with_odds.csv"
OUT = "data/focused_ticket_rule_simulation.csv"


def fmt_pct(x):
    return f"{x:.2%}"


def main():
    print("loading...")
    df = pd.read_csv(IN, dtype={"race_id": str, "combination": str})

    df["race_score"] = pd.to_numeric(df["race_ai_score"], errors="coerce").fillna(0)
    df["ticket_score"] = pd.to_numeric(df["ticket_score"], errors="coerce").fillna(0)
    df["estimated_payout"] = pd.to_numeric(df["estimated_payout"], errors="coerce").fillna(0)
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(0)
    df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)
    df["return_yen"] = pd.to_numeric(df["return_yen"], errors="coerce").fillna(0)

    score_sum = df.groupby("race_id")["ticket_score"].transform("sum")
    df["ticket_prob_norm"] = (df["ticket_score"] / score_sum).fillna(0)
    df["expected_roi"] = df["ticket_prob_norm"] * df["estimated_payout"] / 100.0
    df["value_score"] = df["ticket_score"] * df["estimated_payout"]

    base = df[
        (df["race_score"] >= 0.60)
        & (df["odds"] >= 100)
        & (df["odds"] <= 1000)
        & (df["expected_roi"] >= 1.0)
    ].copy()

    print(f"base races  : {base['race_id'].nunique():,}")
    print(f"base tickets : {len(base):,}")

    rows = []

    metrics = ["expected_roi", "ticket_score", "value_score"]
    top_ns = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]

    for metric in metrics:
        for n in top_ns:
            sel = (
                base.sort_values(["race_id", metric], ascending=[True, False])
                .groupby("race_id", group_keys=False)
                .head(n)
                .copy()
            )

            race_count = sel["race_id"].nunique()
            tickets = len(sel)
            bet = tickets * 100
            ret = int(sel["return_yen"].sum())
            hit = int(sel["is_hit"].sum())
            max_hit = int(sel["return_yen"].max()) if tickets else 0

            rows.append({
                "rule": f"race>=0.60_odds100-1000_ev>=1_top{n}_by_{metric}",
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
                "avg_race_score": sel["race_score"].mean() if tickets else 0,
                "avg_odds": sel["odds"].mean() if tickets else 0,
                "median_odds": sel["odds"].median() if tickets else 0,
                "avg_expected_roi": sel["expected_roi"].mean() if tickets else 0,
                "min_expected_roi": sel["expected_roi"].min() if tickets else 0,
                "avg_hit_payout": sel.loc[sel["is_hit"].eq(1), "return_yen"].mean() if hit else 0,
                "max_hit_payout": max_hit,
                "max_hit_share": max_hit / ret if ret else 0,
            })

    out = pd.DataFrame(rows).sort_values(
        ["roi_without_max_hit", "roi", "hit_count"],
        ascending=[False, False, False],
    )

    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT}")

    print(out.to_string(
        index=False,
        formatters={
            "roi": fmt_pct,
            "roi_without_max_hit": fmt_pct,
            "hit_rate_per_ticket": fmt_pct,
            "hit_rate_per_race": fmt_pct,
            "max_hit_share": fmt_pct,
            "avg_tickets_per_race": lambda x: f"{x:.1f}",
            "avg_race_score": lambda x: f"{x:.4f}",
            "avg_odds": lambda x: f"{x:.1f}",
            "median_odds": lambda x: f"{x:.1f}",
            "avg_expected_roi": lambda x: f"{x:.2f}",
            "min_expected_roi": lambda x: f"{x:.2f}",
            "avg_hit_payout": lambda x: f"{x:.0f}",
            "max_hit_payout": lambda x: f"{x:.0f}",
        },
    ))


if __name__ == "__main__":
    main()
