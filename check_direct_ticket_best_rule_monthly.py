import pandas as pd

df = pd.read_csv("data/direct_ticket_predictions_full_valid.csv", dtype={"race_id": str})

df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
df["month"] = df["race_date"].dt.to_period("M").astype(str)

df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(0)
df["direct_ticket_score"] = pd.to_numeric(df["direct_ticket_score"], errors="coerce").fillna(0)
df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)
df["return_yen"] = pd.to_numeric(df["return_yen"], errors="coerce").fillna(0)

base = df[
    (df["odds"] >= 30)
    & (df["odds"] <= 500)
    & (df["direct_ticket_score"] >= 0.9)
].copy()

sel = (
    base.sort_values(["race_id", "direct_ticket_score"], ascending=[True, False])
    .groupby("race_id", group_keys=False)
    .head(2)
)

rows = []
for month, g in sel.groupby("month"):
    bet = len(g) * 100
    ret = int(g["return_yen"].sum())
    hit = int(g["is_hit"].sum())
    max_hit = int(g["return_yen"].max()) if len(g) else 0

    rows.append({
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
out.to_csv("data/direct_ticket_best_rule_monthly.csv", index=False, encoding="utf-8-sig")

print(out.to_string(
    index=False,
    formatters={
        "roi": lambda x: f"{x:.2%}",
        "roi_without_max_hit": lambda x: f"{x:.2%}",
    }
))
