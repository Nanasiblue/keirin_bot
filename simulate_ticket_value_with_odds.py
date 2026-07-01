from __future__ import annotations

from pathlib import Path

import pandas as pd


TICKETS = Path("data/ticket_scores_valid.csv")
ODDS = Path("data/odds_3rentan_all.csv")
PAYOUTS = Path("data/payouts_all_kdreams.csv")

OUT_JOINED = Path("data/ticket_scores_valid_with_odds.csv")
OUT_SUMMARY = Path("data/ticket_value_simulation.csv")


def pick_score_col(df: pd.DataFrame) -> str:
    for col in ["ticket_score", "score", "pred_score"]:
        if col in df.columns:
            return col
    raise ValueError(f"ticket score column not found. columns={df.columns.tolist()}")


def make_key(df: pd.DataFrame) -> pd.Series:
    return df["race_id"].astype(str) + "_" + df["combination"].astype(str)


def load_filtered_odds(target_keys: set[str]) -> pd.DataFrame:
    chunks = []
    total = 0
    matched = 0

    usecols = ["race_id", "combination", "odds", "estimated_payout"]

    for chunk in pd.read_csv(
        ODDS,
        usecols=usecols,
        dtype={"race_id": str, "combination": str},
        chunksize=1_000_000,
    ):
        total += len(chunk)
        key = make_key(chunk)
        hit = chunk[key.isin(target_keys)].copy()
        if len(hit):
            chunks.append(hit)
            matched += len(hit)

        print(f"odds scanned: {total:,} / matched: {matched:,}")

    if not chunks:
        return pd.DataFrame(columns=usecols)

    return pd.concat(chunks, ignore_index=True)


def simulate_rule(df: pd.DataFrame, rule_name: str, selected: pd.DataFrame) -> dict:
    race_count = selected["race_id"].nunique()
    tickets = len(selected)
    bet = tickets * 100
    ret = int(selected["return_yen"].sum())
    profit = ret - bet
    roi = ret / bet if bet else 0
    hit_count = int(selected["is_hit"].sum())

    return {
        "rule": rule_name,
        "race_count": race_count,
        "tickets": tickets,
        "bet": bet,
        "return": ret,
        "profit": profit,
        "roi": roi,
        "hit_count": hit_count,
        "hit_rate_per_ticket": hit_count / tickets if tickets else 0,
        "hit_rate_per_race": hit_count / race_count if race_count else 0,
        "avg_odds": selected["odds"].mean() if tickets else 0,
        "median_odds": selected["odds"].median() if tickets else 0,
        "avg_value_score": selected["value_score"].mean() if tickets else 0,
        "avg_hit_payout": selected.loc[selected["is_hit"] == 1, "return_yen"].mean() if hit_count else 0,
        "max_hit_payout": selected["return_yen"].max() if tickets else 0,
    }


def top_per_race(df: pd.DataFrame, metric: str, n: int) -> pd.DataFrame:
    return (
        df.sort_values(["race_id", metric], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(n)
    )


def main():
    print("loading tickets...")
    tickets = pd.read_csv(TICKETS, dtype={"race_id": str, "combination": str})

    if "race_id" not in tickets.columns or "combination" not in tickets.columns:
        raise ValueError(f"ticket_scores_valid.csv must have race_id and combination. columns={tickets.columns.tolist()}")

    score_col = pick_score_col(tickets)
    print(f"score column: {score_col}")
    print(f"tickets: {len(tickets):,}")
    print(f"races: {tickets['race_id'].nunique():,}")

    target_keys = set(make_key(tickets))
    print(f"target keys: {len(target_keys):,}")

    print("loading matched odds from big odds csv...")
    odds = load_filtered_odds(target_keys)
    print(f"matched odds rows: {len(odds):,}")

    print("joining odds...")
    df = tickets.merge(odds, on=["race_id", "combination"], how="left")

    missing_odds = df["odds"].isna().sum()
    print(f"missing odds: {missing_odds:,}")

    df = df.dropna(subset=["odds"]).copy()
    df["estimated_payout"] = df["estimated_payout"].astype(float)

    if "is_hit" in df.columns and "return_yen" in df.columns:
        print("using existing hit labels from ticket_scores_valid.csv...")
        df["is_hit"] = df["is_hit"].fillna(0).astype(int)
        df["return_yen"] = df["return_yen"].fillna(0)
    else:
        print("loading payouts...")
        payouts = pd.read_csv(
            PAYOUTS,
            dtype={"race_id": str, "combination": str},
        )

        payouts = payouts[payouts["bet_type"].astype(str).eq("3連単")].copy()
        payouts = payouts[["race_id", "combination", "payout", "popularity"]].rename(
            columns={
                "combination": "actual_hit_combination",
                "payout": "actual_payout",
                "popularity": "actual_popularity",
            }
        )

        df = df.merge(payouts, on="race_id", how="left")
        df["is_hit"] = (df["combination"] == df["actual_hit_combination"]).astype(int)
        df["return_yen"] = df["is_hit"] * df["actual_payout"].fillna(0)

    score_sum = df.groupby("race_id")[score_col].transform("sum")
    df["ticket_prob_norm"] = (df[score_col] / score_sum).fillna(0)

    # 100円賭けの期待回収率。1.0なら理論上トントン相当
    df["expected_roi"] = df["ticket_prob_norm"] * df["estimated_payout"] / 100.0

    df["value_score"] = df[score_col] * df["estimated_payout"]

    if "race_score" not in df.columns:
        df["race_score"] = 0.0

    OUT_JOINED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_JOINED, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_JOINED}")
    print(f"joined rows: {len(df):,}")

    race_thresholds = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    odds_ranges = [
        ("odds_all", 0, 99999),
        ("odds_30plus", 30, 99999),
        ("odds_50plus", 50, 99999),
        ("odds_100plus", 100, 99999),
        ("odds_50_500", 50, 500),
        ("odds_100_1000", 100, 1000),
    ]
    metrics = [score_col, "value_score", "expected_roi"]
    top_ns = [1, 3, 5, 10, 20, 30]

    rows = []

    print("simulating...")
    for race_th in race_thresholds:
        base = df[df["race_score"] >= race_th].copy()

        for odds_name, lo, hi in odds_ranges:
            odds_base = base[(base["odds"] >= lo) & (base["odds"] <= hi)].copy()

            if odds_base.empty:
                continue

            for metric in metrics:
                for n in top_ns:
                    selected = top_per_race(odds_base, metric, n)
                    rule = f"race_score>={race_th:.2f}_{odds_name}_top{n}_by_{metric}"
                    rows.append(simulate_rule(df, rule, selected))

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["roi", "return", "tickets"], ascending=[False, False, False])

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT_SUMMARY}")
    print(
        summary.head(40).to_string(
            index=False,
            formatters={
                "roi": lambda x: f"{x:.2%}",
                "hit_rate_per_ticket": lambda x: f"{x:.2%}",
                "hit_rate_per_race": lambda x: f"{x:.2%}",
                "avg_odds": lambda x: f"{x:.1f}",
                "median_odds": lambda x: f"{x:.1f}",
                "avg_value_score": lambda x: f"{x:.1f}",
                "avg_hit_payout": lambda x: f"{x:.0f}",
                "max_hit_payout": lambda x: f"{x:.0f}",
            },
        )
    )


if __name__ == "__main__":
    main()


