import pandas as pd

df = pd.read_csv("data/ticket_scores_valid_with_odds.csv", dtype={"race_id": str})

df["start_date"] = pd.to_datetime(df["race_id"].str[2:10], format="%Y%m%d", errors="coerce")
df["day_no"] = pd.to_numeric(df["race_id"].str[10:12], errors="coerce").fillna(1).astype(int)
df["race_date"] = df["start_date"] + pd.to_timedelta(df["day_no"] - 1, unit="D")
df["month"] = df["race_date"].dt.to_period("M").astype(str)

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

def topn(n):
    return (
        base.sort_values(["race_id", "expected_roi"], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(n)
    )

targets = {
    "all_ev>=1": base,
    "top10": topn(10),
    "top20": topn(20),
    "top50": topn(50),
    "top100": topn(100),
}

rows = []
for name, sel in targets.items():
    for month, g in sel.groupby("month"):
        bet = len(g) * 100
        ret = int(g["return_yen"].sum())
        hit = int(g["is_hit"].sum())
        max_hit = int(g["return_yen"].max()) if len(g) else 0
        rows.append({
            "rule": name,
            "month": month,
            "races": g["race_id"].nunique(),
            "tickets": len(g),
            "bet": bet,
            "return": ret,
            "profit": ret - bet,
            "roi": ret / bet if bet else 0,
            "roi_without_max_hit": (ret - max_hit) / bet if bet else 0,
            "hit_count": hit,
            "max_hit": max_hit,
        })

out = pd.DataFrame(rows)
out.to_csv("data/focused_rule_monthly.csv", index=False, encoding="utf-8-sig")

print(out.to_string(
    index=False,
    formatters={
        "roi": lambda x: f"{x:.2%}",
        "roi_without_max_hit": lambda x: f"{x:.2%}",
    }
))
