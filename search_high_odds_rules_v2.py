import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRED_IN = Path("data/direct_ticket_predictions_full_valid.csv")
RACE_SCORE_IN = Path("data/predictions_valid_2026_is_over_50.csv")
PAYOUT_IN = Path("data/payouts_all_kdreams.csv")
OUT = Path("data/high_odds_rule_search_v2.csv")
OUT_MONTHLY = Path("data/high_odds_rule_search_v2_monthly.csv")

LEAK_COLUMNS = {
    "is_over_30", "is_over_50", "is_over_100",
    "payout_3rentan", "payout_popularity", "payout",
    "popularity", "combination", "target", "label", "actual",
    "hit", "is_hit", "return"
}

PREFERRED_RACE_SCORE_COLS = [
    "race_score",
    "race_ai_score",
    "pred_score",
    "prediction_score",
    "pred_proba",
    "prob",
    "proba",
    "probability",
    "model_score",
    "rf_score",
    "score",
]


def pct(x):
    return f"{x * 100:.2f}%"


def first_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def detect_race_score_col(df, explicit=None):
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"--race-score-col が見つかりません: {explicit}")
        if explicit in LEAK_COLUMNS or explicit.startswith("is_over_"):
            raise SystemExit(f"危険: {explicit} は正解/結果列っぽいので使いません")
        return explicit

    safe_numeric = []
    for col in df.columns:
        low = col.lower()
        if col == "race_id":
            continue
        if col in LEAK_COLUMNS or col.startswith("is_over_"):
            continue
        if "payout" in low or "popularity" in low or "combination" in low:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            safe_numeric.append(col)

    for col in PREFERRED_RACE_SCORE_COLS:
        if col in safe_numeric:
            return col

    score_like = [
        c for c in safe_numeric
        if any(k in c.lower() for k in ["score", "pred", "prob", "proba"])
    ]

    if len(score_like) == 1:
        return score_like[0]

    print("race score列を自動判定できませんでした。候補:")
    for c in score_like or safe_numeric:
        s = df[c]
        print(f"  {c}: min={s.min()} max={s.max()} nunique={s.nunique()}")
    print("")
    print("例:")
    print("  python search_high_odds_rules_v2.py --race-score-col 予測スコア列名")
    raise SystemExit(1)


def summarize_rule(df, rule_name):
    if df.empty:
        return None

    bet = len(df) * 100
    ret = int(df.loc[df["is_hit"] == 1, "hit_payout"].sum())
    profit = ret - bet
    hit_count = int(df["is_hit"].sum())
    max_hit = int(df.loc[df["is_hit"] == 1, "hit_payout"].max()) if hit_count else 0
    ret_without_max = ret - max_hit
    race_count = df["race_id"].nunique()

    return {
        "rule": rule_name,
        "race_count": race_count,
        "tickets": len(df),
        "avg_tickets_per_race": len(df) / race_count if race_count else 0,
        "bet": bet,
        "return": ret,
        "profit": profit,
        "roi": ret / bet if bet else 0,
        "roi_without_max_hit": ret_without_max / bet if bet else 0,
        "hit_count": hit_count,
        "hit_rate_per_ticket": hit_count / len(df) if len(df) else 0,
        "hit_rate_per_race": hit_count / race_count if race_count else 0,
        "avg_race_score": df["race_score"].mean(),
        "min_race_score": df["race_score"].min(),
        "avg_ticket_score": df["ticket_score"].mean(),
        "min_ticket_score": df["ticket_score"].min(),
        "avg_odds": df["odds"].mean(),
        "median_odds": df["odds"].median(),
        "avg_expected": df["expected_return"].mean(),
        "max_hit_payout": max_hit,
        "max_hit_share": max_hit / ret if ret else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race-score-col", default=None)
    args = ap.parse_args()

    print("loading ticket predictions...")
    df = pd.read_csv(PRED_IN, dtype={"race_id": str})
    print(f"tickets: {len(df):,} / races: {df['race_id'].nunique():,}")

    ticket_score_col = first_col(df, ["direct_ticket_score", "ticket_score", "score", "pred_score", "prob"])
    if ticket_score_col is None:
        raise SystemExit("ticket score列が見つかりません")

    if ticket_score_col != "ticket_score":
        df = df.rename(columns={ticket_score_col: "ticket_score"})

    if "odds" not in df.columns:
        raise SystemExit("odds列が見つかりません")

    if "is_hit" not in df.columns:
        if "hit_combination" in df.columns:
            df["is_hit"] = (df["combination"] == df["hit_combination"]).astype(int)
        else:
            print("is_hitがないのでpayoutsから作ります")

    print("loading race score...")
    race = pd.read_csv(RACE_SCORE_IN, dtype={"race_id": str})
    race_col = detect_race_score_col(race, args.race_score_col)

    if race_col in LEAK_COLUMNS or race_col.startswith("is_over_"):
        raise SystemExit(f"危険: {race_col} は使いません")

    uniq = set(pd.Series(race[race_col].dropna().unique()).head(5).tolist())
    if race[race_col].nunique() <= 2 and uniq.issubset({0, 1, 0.0, 1.0}):
        raise SystemExit(
            f"危険: {race_col} は0/1だけです。正解ラベルの可能性が高いので止めます。"
        )

    race = race[["race_id", race_col]].rename(columns={race_col: "race_score"})
    print(f"race score column: {race_col}")

    df = df.merge(race, on="race_id", how="left")
    missing = df["race_score"].isna().sum()
    print(f"race_score missing: {missing:,}")
    df = df.dropna(subset=["race_score"]).copy()

    print("loading payouts...")
    pay = pd.read_csv(PAYOUT_IN, dtype={"race_id": str})
    pay = pay.rename(columns={"combination": "hit_combination", "payout": "hit_payout"})
    pay = pay[["race_id", "hit_combination", "hit_payout"]]
    df = df.merge(pay, on="race_id", how="left", suffixes=("", "_official"))

    if "hit_combination_official" in df.columns:
        if "hit_combination" not in df.columns:
            df["hit_combination"] = df["hit_combination_official"]
        else:
            df["hit_combination"] = df["hit_combination"].fillna(df["hit_combination_official"])

    if "hit_payout_official" in df.columns:
        if "hit_payout" not in df.columns:
            df["hit_payout"] = df["hit_payout_official"]
        else:
            df["hit_payout"] = df["hit_payout"].fillna(df["hit_payout_official"])

    if "is_hit" not in df.columns:
        df["is_hit"] = (df["combination"] == df["hit_combination"]).astype(int)

    if "hit_payout" not in df.columns:
        payout_cols = [c for c in df.columns if "payout" in c.lower()]
        raise SystemExit(f"hit_payout列が作れませんでした。payout系の列: {payout_cols}")

    df["hit_payout"] = np.where(df["is_hit"] == 1, df["hit_payout"], 0)
    df["expected_return"] = df["ticket_score"] * df["odds"] * 100

    print(f"rows after merge: {len(df):,}")
    print(f"race_score min/max: {df['race_score'].min():.4f} / {df['race_score'].max():.4f}")

    odds_ranges = [
        ("odds_100_500", 100, 500),
        ("odds_100_1000", 100, 1000),
        ("odds_200_1000", 200, 1000),
        ("odds_200_2000", 200, 2000),
        ("odds_300_3000", 300, 3000),
    ]
    race_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    score_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    expected_thresholds = [3000, 5000, 10000, 20000]
    top_ns = [1, 2, 3, 5, 10]

    rows = []
    monthly_rows = []

    print("searching...")
    for odds_name, odds_min, odds_max in odds_ranges:
        base_odds = df[(df["odds"] >= odds_min) & (df["odds"] <= odds_max)].copy()

        for race_th in race_thresholds:
            base_race = base_odds[base_odds["race_score"] >= race_th].copy()

            for score_th in score_thresholds:
                base_score = base_race[base_race["ticket_score"] >= score_th].copy()

                for exp_th in expected_thresholds:
                    base_exp = base_score[base_score["expected_return"] >= exp_th].copy()

                    for sort_col in ["ticket_score", "expected_return"]:
                        sorted_df = base_exp.sort_values(
                            ["race_id", sort_col],
                            ascending=[True, False],
                        )

                        for top_n in top_ns:
                            picked = sorted_df.groupby("race_id", as_index=False).head(top_n)
                            rule = (
                                f"{odds_name}_race>={race_th}_score>={score_th}"
                                f"_expected>={exp_th}_top{top_n}_by_{sort_col}"
                            )
                            s = summarize_rule(picked, rule)
                            if s is None:
                                continue

                            mlist = []
                            if "race_date" in picked.columns:
                                tmp = picked.copy()
                                tmp["month"] = pd.to_datetime(tmp["race_date"], errors="coerce").dt.strftime("%Y-%m")
                                for month, g in tmp.dropna(subset=["month"]).groupby("month"):
                                    ms = summarize_rule(g, rule)
                                    if ms:
                                        ms["month"] = month
                                        monthly_rows.append(ms)
                                        mlist.append(ms)

                            if mlist:
                                month_rois = [x["roi"] for x in mlist]
                                s["month_min_roi"] = min(month_rois)
                                s["month_mean_roi"] = sum(month_rois) / len(month_rois)
                                s["month_count"] = len(month_rois)
                                s["plus_months"] = sum(x >= 1.0 for x in month_rois)
                            else:
                                s["month_min_roi"] = 0
                                s["month_mean_roi"] = 0
                                s["month_count"] = 0
                                s["plus_months"] = 0

                            s["enough_hits"] = s["hit_count"] >= 20
                            rows.append(s)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["enough_hits", "roi_without_max_hit", "roi", "hit_count"],
        ascending=[False, False, False, False],
    )

    display = out.copy()
    for c in ["roi", "roi_without_max_hit", "hit_rate_per_ticket", "hit_rate_per_race",
              "max_hit_share", "month_min_roi", "month_mean_roi"]:
        display[c] = display[c].map(pct)
    for c in ["avg_tickets_per_race", "avg_race_score", "min_race_score",
              "avg_ticket_score", "min_ticket_score", "avg_odds", "median_odds", "avg_expected"]:
        display[c] = display[c].map(lambda x: f"{x:.4f}" if "score" in c else f"{x:.1f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    monthly = pd.DataFrame(monthly_rows)
    if not monthly.empty:
        monthly.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT}")
    print(f"saved: {OUT_MONTHLY}")
    print(display.head(50).to_string(index=False))


if __name__ == "__main__":
    main()

