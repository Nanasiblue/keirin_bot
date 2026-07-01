from __future__ import annotations

from pathlib import Path
import pandas as pd

TICKETS = Path("data/ticket_scores_valid.csv")
ODDS = Path("data/odds_3rentan_all.csv")
OUT_JOINED = Path("data/ticket_scores_valid_with_odds.csv")
OUT_SUMMARY = Path("data/ticket_value_simulation.csv")


def pick_score_col(df):
    for col in ["ticket_score", "score", "pred_score"]:
        if col in df.columns:
            return col
    raise ValueError(f"ticket score column not found: {df.columns.tolist()}")


def make_key(df):
    return df["race_id"].astype(str) + "_" + df["combination"].astype(str)


def load_filtered_odds(target_keys):
    chunks = []
    total = 0
    matched = 0

    for chunk in pd.read_csv(
        ODDS,
        usecols=["race_id", "combination", "odds", "estimated_payout"],
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

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def build_joined():
    print("loading tickets...")
    tickets = pd.read_csv(TICKETS, dtype={"race_id": str, "combination": str})
    print(f"tickets: {len(tickets):,}")
    print(f"races: {tickets['race_id'].nunique():,}")

    target_keys = set(make_key(tickets))
    print(f"target keys: {len(target_keys):,}")

    print("loading matched odds from big odds csv...")
    odds = load_filtered_odds(target_keys)
    print(f"matched odds rows: {len(odds):,}")

    print("joining odds...")
    df = tickets.merge(odds, on=["race_id", "combination"], how="left")
    print(f"missing odds: {df['odds'].isna().sum():,}")

    df = df.dropna(subset=["odds"]).copy()
    df.to_csv(OUT_JOINED, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_JOINED}")
    return df


def load_joined():
    if OUT_JOINED.exists():
        print(f"loading existing joined file: {OUT_JOINED}")
        return pd.read_csv(OUT_JOINED, dtype={"race_id": str, "combination": str})
    return build_joined()


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
    ret = int(selected["return_yen"].sum()) if tickets else 0
    hit = int(selected["is_hit"].sum()) if tickets else 0

    return {
        "rule": rule,
        "race_count": race_count,
        "tickets": tickets,
        "bet": bet,
        "return": ret,
        "profit": ret - bet,
        "roi": ret / bet if bet else 0,
        "hit_count": hit,
        "hit_rate_per_ticket": hit / tickets if tickets else 0,
        "hit_rate_per_race": hit / race_count if race_count else 0,
        "avg_race_score": selected["race_score"].mean() if tickets else 0,
        "min_race_score": selected["race_score"].min() if tickets else 0,
        "avg_odds": selected["odds"].mean() if tickets else 0,
        "median_odds": selected["odds"].median() if tickets else 0,
        "avg_expected_roi": selected["expected_roi"].mean() if tickets else 0,
        "min_expected_roi": selected["expected_roi"].min() if tickets else 0,
        "avg_hit_payout": selected.loc[selected["is_hit"].eq(1), "return_yen"].mean() if hit else 0,
        "max_hit_payout": selected["return_yen"].max() if tickets else 0,
    }


def main():
    df = load_joined()
    score_col = pick_score_col(df)

    for col in [score_col, "estimated_payout", "odds", "return_yen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)

    # ここが修正点。荒れレースAIのスコアをちゃんと使う。
    if "race_ai_score" in df.columns:
        df["race_score"] = pd.to_numeric(df["race_ai_score"], errors="coerce").fillna(0)
    elif "race_score" in df.columns:
        df["race_score"] = pd.to_numeric(df["race_score"], errors="coerce").fillna(0)
    else:
        df["race_score"] = 0.0

    score_sum = df.groupby("race_id")[score_col].transform("sum")
    df["ticket_prob_norm"] = (df[score_col] / score_sum).fillna(0)
    df["expected_roi"] = df["ticket_prob_norm"] * df["estimated_payout"] / 100.0
    df["value_score"] = df[score_col] * df["estimated_payout"]

    print(f"rows: {len(df):,}")
    print(f"races: {df['race_id'].nunique():,}")
    print(f"score column: {score_col}")
    print(f"race_score min/max: {df['race_score'].min():.4f} / {df['race_score'].max():.4f}")

    # 更新済みの列を保存しておく
    df.to_csv(OUT_JOINED, index=False, encoding="utf-8-sig")
    print(f"saved refreshed joined file: {OUT_JOINED}")

    race_thresholds = [0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    odds_ranges = [
        ("odds_all", 0, 99999),
        ("odds_30plus", 30, 99999),
        ("odds_50plus", 50, 99999),
        ("odds_100plus", 100, 99999),
        ("odds_30_500", 30, 500),
        ("odds_50_500", 50, 500),
        ("odds_50_1000", 50, 1000),
        ("odds_100_1000", 100, 1000),
    ]
    metrics = [score_col, "value_score", "expected_roi"]
    top_ns = [1, 2, 3, 5, 10, 20, 30]
    ev_thresholds = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

    rows = []

    print("simulating...")
    for race_th in race_thresholds:
        base = df[df["race_score"] >= race_th].copy()
        if base.empty:
            continue

        for odds_name, lo, hi in odds_ranges:
            odds_base = base[(base["odds"] >= lo) & (base["odds"] <= hi)].copy()
            if odds_base.empty:
                continue

            for metric in metrics:
                for n in top_ns:
                    selected = top_per_race(odds_base, metric, n)
                    rule = f"race_score>={race_th:.2f}_{odds_name}_top{n}_by_{metric}"
                    rows.append(simulate(selected, rule))

            for ev_th in ev_thresholds:
                ev_base = odds_base[odds_base["expected_roi"] >= ev_th].copy()
                if ev_base.empty:
                    continue

                rows.append(simulate(ev_base, f"race_score>={race_th:.2f}_{odds_name}_ev>={ev_th}"))

                for n in [1, 2, 3, 5]:
                    selected = top_per_race(ev_base, "expected_roi", n)
                    rows.append(simulate(selected, f"race_score>={race_th:.2f}_{odds_name}_ev>={ev_th}_top{n}"))

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["roi", "race_count", "tickets"], ascending=[False, False, False])
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT_SUMMARY}")
    print(
        summary.head(60).to_string(
            index=False,
            formatters={
                "roi": lambda x: f"{x:.2%}",
                "hit_rate_per_ticket": lambda x: f"{x:.2%}",
                "hit_rate_per_race": lambda x: f"{x:.2%}",
                "avg_race_score": lambda x: f"{x:.4f}",
                "min_race_score": lambda x: f"{x:.4f}",
                "avg_odds": lambda x: f"{x:.1f}",
                "median_odds": lambda x: f"{x:.1f}",
                "avg_expected_roi": lambda x: f"{x:.2f}",
                "min_expected_roi": lambda x: f"{x:.2f}",
                "avg_hit_payout": lambda x: f"{x:.0f}",
                "max_hit_payout": lambda x: f"{x:.0f}",
            },
        )
    )


if __name__ == "__main__":
    main()
