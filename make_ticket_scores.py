from itertools import permutations
from pathlib import Path
import pandas as pd

DATA = Path("data")
PRED = DATA / "finish_class_predictions_valid.csv"
RACE_PRED = DATA / "predictions_valid_2026_is_over_50.csv"
PAYOUTS = DATA / "payouts_all_kdreams.csv"

OUT_TICKETS = DATA / "ticket_scores_valid.csv"
OUT_SIM = DATA / "ticket_score_simulation.csv"

BET_UNIT = 100

def main():
    riders = pd.read_csv(PRED, dtype={"race_id": str})
    races = pd.read_csv(RACE_PRED, dtype={"race_id": str})
    payouts = pd.read_csv(PAYOUTS, dtype={"race_id": str})

    riders["race_id"] = riders["race_id"].astype(str).str.zfill(16)
    races["race_id"] = races["race_id"].astype(str).str.zfill(16)
    payouts["race_id"] = payouts["race_id"].astype(str).str.zfill(16)

    races = races[["race_id", "ai_score"]].rename(columns={"ai_score": "race_ai_score"})
    pay = payouts[payouts["bet_type"].astype(str).str.contains("3連単", na=False)].copy()
    pay = pay[["race_id", "combination", "payout", "popularity"]].drop_duplicates("race_id")
    pay = pay.rename(columns={
        "combination": "hit_combination",
        "payout": "hit_payout",
        "popularity": "hit_popularity",
    })

    riders = riders.merge(races, on="race_id", how="left")
    riders = riders.dropna(subset=["race_ai_score"]).copy()

    rows = []
    for race_id, g in riders.groupby("race_id"):
        g = g.copy()
        race_score = float(g["race_ai_score"].iloc[0])
        car_map = {
            int(row.car_no): {
                "name": row.name,
                "p_1st": float(row.p_1st),
                "p_2nd": float(row.p_2nd),
                "p_3rd": float(row.p_3rd),
                "p_4plus": float(row.p_4plus),
                "p_exception": float(row.p_exception),
                "p_top3": float(row.p_top3),
            }
            for row in g.itertuples()
        }

        for a, b, c in permutations(car_map.keys(), 3):
            r1 = car_map[a]
            r2 = car_map[b]
            r3 = car_map[c]
            rank_score = r1["p_1st"] * r2["p_2nd"] * r3["p_3rd"]
            ticket_score = rank_score * race_score
            rows.append({
                "race_id": race_id,
                "combination": f"{a}-{b}-{c}",
                "ticket_score": ticket_score,
                "rank_score": rank_score,
                "race_ai_score": race_score,
                "first_car": a,
                "second_car": b,
                "third_car": c,
                "first_name": r1["name"],
                "second_name": r2["name"],
                "third_name": r3["name"],
                "p1": r1["p_1st"],
                "p2": r2["p_2nd"],
                "p3": r3["p_3rd"],
                "first_p_exception": r1["p_exception"],
                "second_p_exception": r2["p_exception"],
                "third_p_exception": r3["p_exception"],
            })

    tickets = pd.DataFrame(rows)
    tickets = tickets.merge(pay, on="race_id", how="left")
    tickets["is_hit"] = (tickets["combination"] == tickets["hit_combination"]).astype(int)
    tickets["return_yen"] = tickets["is_hit"] * tickets["hit_payout"].fillna(0)

    tickets = tickets.sort_values(["race_id", "ticket_score"], ascending=[True, False])
    tickets["score_rank_in_race"] = tickets.groupby("race_id").cumcount() + 1

    OUT_TICKETS.parent.mkdir(parents=True, exist_ok=True)
    tickets.to_csv(OUT_TICKETS, index=False, encoding="utf-8-sig")

    sim_rows = []
    race_count = tickets["race_id"].nunique()

    for top_n in [3, 5, 10, 20, 30, 50, 100]:
        selected = tickets[tickets["score_rank_in_race"] <= top_n].copy()
        bet = len(selected) * BET_UNIT
        ret = selected["return_yen"].sum()
        hit_count = int(selected["is_hit"].sum())
        sim_rows.append({
            "rule": f"top_{top_n}_per_race",
            "race_count": race_count,
            "tickets": len(selected),
            "bet": int(bet),
            "return": int(ret),
            "profit": int(ret - bet),
            "roi": ret / bet if bet else 0,
            "hit_count": hit_count,
            "hit_rate_per_race": hit_count / race_count if race_count else 0,
            "avg_hit_payout": selected.loc[selected["is_hit"] == 1, "hit_payout"].mean(),
            "max_hit_payout": selected.loc[selected["is_hit"] == 1, "hit_payout"].max(),
        })

    # 荒れAIが高いレースだけ買う版
    for race_th in [0.50, 0.55, 0.60, 0.65]:
        sub = tickets[tickets["race_ai_score"] >= race_th]
        rc = sub["race_id"].nunique()
        for top_n in [10, 20, 30, 50]:
            selected = sub[sub["score_rank_in_race"] <= top_n].copy()
            bet = len(selected) * BET_UNIT
            ret = selected["return_yen"].sum()
            hit_count = int(selected["is_hit"].sum())
            sim_rows.append({
                "rule": f"race_score>={race_th:.2f}_top_{top_n}",
                "race_count": rc,
                "tickets": len(selected),
                "bet": int(bet),
                "return": int(ret),
                "profit": int(ret - bet),
                "roi": ret / bet if bet else 0,
                "hit_count": hit_count,
                "hit_rate_per_race": hit_count / rc if rc else 0,
                "avg_hit_payout": selected.loc[selected["is_hit"] == 1, "hit_payout"].mean(),
                "max_hit_payout": selected.loc[selected["is_hit"] == 1, "hit_payout"].max(),
            })

    sim = pd.DataFrame(sim_rows).sort_values("roi", ascending=False)
    sim.to_csv(OUT_SIM, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT_TICKETS}")
    print(f"saved: {OUT_SIM}")
    print("")
    print(sim.to_string(index=False, formatters={
        "roi": "{:.2%}".format,
        "hit_rate_per_race": "{:.2%}".format,
        "avg_hit_payout": "{:.0f}".format,
        "max_hit_payout": "{:.0f}".format,
    }))

if __name__ == "__main__":
    main()
