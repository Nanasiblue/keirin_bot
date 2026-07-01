from pathlib import Path
import pandas as pd

MIDDLE = "data/fixed_rule_direct_v1_middle_odds.csv"
HIGH = "data/fixed_rule_high_odds_v1.csv"

def pct(x): return f"{x:.2%}"

def norm(df, rule=None):
    df = df.copy()
    if "rule" not in df.columns:
        df.insert(0, "rule", rule)
    elif rule:
        df["rule"] = rule
    df["race_id"] = df["race_id"].astype(str)
    df["combination"] = df["combination"].astype(str)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["month"] = df["race_date"].dt.to_period("M").astype(str)
    for c in ["odds", "return_yen", "direct_ticket_score", "race_score", "expected_return"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0
    df["is_hit"] = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)
    return df

def stake(df, strategy, yen):
    x = df.copy()
    x["strategy"] = strategy
    x["stake_yen"] = yen
    x["scaled_return_yen"] = x["return_yen"] * (yen / 100)
    x["profit_yen"] = x["scaled_return_yen"] - yen
    return x

def top_expected(df, n):
    return df.sort_values(["race_id", "expected_return"], ascending=[True, False]).groupby("race_id").head(n).copy()

def equity(df):
    e = df.groupby(["strategy", "race_date", "race_id"], as_index=False).agg(
        tickets=("combination", "size"),
        bet=("stake_yen", "sum"),
        ret=("scaled_return_yen", "sum"),
        hits=("is_hit", "sum"),
    ).sort_values(["strategy", "race_date", "race_id"])
    e["profit"] = e["ret"] - e["bet"]
    e["cum_profit"] = e.groupby("strategy")["profit"].cumsum()
    e["peak"] = e.groupby("strategy")["cum_profit"].cummax()
    e["drawdown"] = e["cum_profit"] - e["peak"]
    return e

def losing_streak(e):
    worst = cur = 0
    for _, r in e.sort_values(["race_date", "race_id"]).iterrows():
        if r["profit"] < 0:
            cur += 1
            worst = max(worst, cur)
        else:
            cur = 0
    return worst

def summary(t, e):
    rows = []
    for s, g in t.groupby("strategy"):
        eg = e[e["strategy"] == s]
        bet = g["stake_yen"].sum()
        ret = g["scaled_return_yen"].sum()
        max_hit = g["scaled_return_yen"].max() if len(g) else 0
        races = g["race_id"].nunique()
        rows.append({
            "strategy": s,
            "races": races,
            "tickets": len(g),
            "avg_tickets": len(g) / races if races else 0,
            "bet": int(bet),
            "return": int(ret),
            "profit": int(ret - bet),
            "roi": ret / bet if bet else 0,
            "roi_wo_max": (ret - max_hit) / bet if bet else 0,
            "hits": int(g["is_hit"].sum()),
            "hit_rate": g["is_hit"].mean() if len(g) else 0,
            "avg_bet_race": bet / races if races else 0,
            "max_drawdown": int(eg["drawdown"].min()) if len(eg) else 0,
            "max_losing_streak": losing_streak(eg),
        })
    return pd.DataFrame(rows).sort_values(["roi_wo_max", "roi"], ascending=[False, False])

def monthly(t):
    rows = []
    for (s, m), g in t.groupby(["strategy", "month"]):
        bet = g["stake_yen"].sum()
        ret = g["scaled_return_yen"].sum()
        max_hit = g["scaled_return_yen"].max() if len(g) else 0
        rows.append({
            "strategy": s,
            "month": m,
            "races": g["race_id"].nunique(),
            "tickets": len(g),
            "bet": int(bet),
            "return": int(ret),
            "profit": int(ret - bet),
            "roi": ret / bet if bet else 0,
            "roi_wo_max": (ret - max_hit) / bet if bet else 0,
            "hits": int(g["is_hit"].sum()),
        })
    return pd.DataFrame(rows)

def show(df):
    x = df.copy()
    for c in ["roi", "roi_wo_max", "hit_rate"]:
        if c in x.columns:
            x[c] = x[c].map(pct)
    return x

middle = norm(pd.read_csv(MIDDLE, dtype={"race_id": str, "combination": str}), "direct_v1_middle_odds")
high = norm(pd.read_csv(HIGH, dtype={"race_id": str, "combination": str}))
h5 = high[high["rule"] == "high_odds_v1_top5"].copy()
h2_score = high[high["rule"] == "high_odds_v1_top2"].copy()
h2_exp = top_expected(h5, 2)

parts = []
parts.append(stake(middle, "A_middle_only_100", 100))
parts += [stake(middle, "B_middle100_high50_top5", 100), stake(h5, "B_middle100_high50_top5", 50)]
parts += [stake(middle, "C_middle100_high100_top5", 100), stake(h5, "C_middle100_high100_top5", 100)]
for th in [0.50, 0.55, 0.60]:
    name = f"D_middle100_high100_top5_race>={th:.2f}"
    parts += [stake(middle, name, 100), stake(h5[h5["race_score"] >= th], name, 100)]
parts += [stake(middle, "E_middle100_high100_top2_expected", 100), stake(h2_exp, "E_middle100_high100_top2_expected", 100)]
parts += [stake(middle, "F_middle100_high100_top2_score", 100), stake(h2_score, "F_middle100_high100_top2_score", 100)]

tickets = pd.concat(parts, ignore_index=True, sort=False)
tickets = tickets.sort_values(["strategy", "race_id", "combination", "stake_yen"], ascending=[True, True, True, False])
tickets = tickets.drop_duplicates(["strategy", "race_id", "combination"], keep="first")

eq = equity(tickets)
sm = summary(tickets, eq)
mo = monthly(tickets)

Path("data").mkdir(exist_ok=True)
tickets.to_csv("data/strategy_allocation_tickets.csv", index=False, encoding="utf-8-sig")
eq.to_csv("data/strategy_allocation_equity.csv", index=False, encoding="utf-8-sig")
sm.to_csv("data/strategy_allocation_summary.csv", index=False, encoding="utf-8-sig")
mo.to_csv("data/strategy_allocation_monthly.csv", index=False, encoding="utf-8-sig")

print("=== allocation summary ===")
print(show(sm).to_string(index=False))

print("")
print("=== monthly ===")
for s in sm["strategy"]:
    print("")
    print("---", s, "---")
    print(show(mo[mo["strategy"] == s]).to_string(index=False))

print("")
print("saved: data/strategy_allocation_summary.csv")
print("saved: data/strategy_allocation_monthly.csv")
print("saved: data/strategy_allocation_equity.csv")
print("saved: data/strategy_allocation_tickets.csv")
