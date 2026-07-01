from pathlib import Path
import pandas as pd

DATA = Path("data")
RESULTS = DATA / "race_results_full.csv"
ENTRIES = DATA / "entries_all_kdreams.csv"
PAYOUTS = DATA / "payouts_all_kdreams.csv"
OUT = DATA / "race_results_full_clean.csv"

def norm_id(s):
    return s.astype(str).str.strip().str.zfill(16)

entries = pd.read_csv(ENTRIES, dtype={"race_id": str})
results = pd.read_csv(RESULTS, dtype={"race_id": str})
payouts = pd.read_csv(PAYOUTS, dtype={"race_id": str})

entries["race_id"] = norm_id(entries["race_id"])
results["race_id"] = norm_id(results["race_id"])
payouts["race_id"] = norm_id(payouts["race_id"])

entries["car_no"] = pd.to_numeric(entries["car_no"], errors="coerce").astype("Int64")
results["car_no"] = pd.to_numeric(results["car_no"], errors="coerce").astype("Int64")
results["finish_position"] = pd.to_numeric(results["finish_position"], errors="coerce").astype("Int64")
results["agari"] = pd.to_numeric(results["agari"], errors="coerce")

result_cols = ["race_id", "car_no", "finish_position", "margin", "agari", "decision", "sb", "comment"]
base = entries.merge(results[result_cols], on=["race_id", "car_no"], how="left")

base["has_numeric_finish"] = base["finish_position"].notna().astype(int)
base["is_finished"] = base["has_numeric_finish"]
base["finish_status"] = "normal"
base.loc[base["has_numeric_finish"] == 0, "finish_status"] = "no_numeric_finish"

finish_counts = (
    base.dropna(subset=["finish_position"])
    .groupby(["race_id", "finish_position"])
    .size()
    .rename("same_finish_count")
    .reset_index()
)
base = base.merge(finish_counts, on=["race_id", "finish_position"], how="left")
base["same_finish_count"] = base["same_finish_count"].fillna(0).astype(int)
base["is_dead_heat"] = (base["same_finish_count"] > 1).astype(int)
base.loc[base["is_dead_heat"] == 1, "finish_status"] = "dead_heat"
base.loc[base["has_numeric_finish"] == 0, "finish_status"] = "no_numeric_finish"

pay = payouts[payouts["bet_type"].astype(str).str.contains("3連単", na=False)].copy()
pay = pay[["race_id", "combination", "payout", "popularity"]].drop_duplicates("race_id")
parts = pay["combination"].astype(str).str.extract(r"^(\d+)-(\d+)-(\d+)$")
pay["ticket_1st_car"] = pd.to_numeric(parts[0], errors="coerce")
pay["ticket_2nd_car"] = pd.to_numeric(parts[1], errors="coerce")
pay["ticket_3rd_car"] = pd.to_numeric(parts[2], errors="coerce")

base = base.merge(pay, on="race_id", how="left")

base["ticket_1st"] = (base["car_no"] == base["ticket_1st_car"]).fillna(False).astype(int)
base["ticket_2nd"] = (base["car_no"] == base["ticket_2nd_car"]).fillna(False).astype(int)
base["ticket_3rd"] = (base["car_no"] == base["ticket_3rd_car"]).fillna(False).astype(int)
base["ticket_top2"] = ((base["ticket_1st"] == 1) | (base["ticket_2nd"] == 1)).astype(int)
base["ticket_top3"] = ((base["ticket_top2"] == 1) | (base["ticket_3rd"] == 1)).astype(int)

base["official_is_1st"] = (base["finish_position"] == 1).fillna(False).astype(int)
base["official_top2"] = (base["finish_position"].le(2) & base["finish_position"].notna()).fillna(False).astype(int)
base["official_top3"] = (base["finish_position"].le(3) & base["finish_position"].notna()).fillna(False).astype(int)

cols = [
    "race_id", "car_no", "name", "score", "style", "age", "win_rate", "place_rate",
    "finish_position", "finish_status", "has_numeric_finish", "is_finished", "is_dead_heat",
    "margin", "agari", "decision", "sb", "comment",
    "combination", "payout", "popularity",
    "ticket_1st", "ticket_2nd", "ticket_3rd", "ticket_top2", "ticket_top3",
    "official_is_1st", "official_top2", "official_top3",
]
cols = [c for c in cols if c in base.columns]
base = base[cols + [c for c in base.columns if c not in cols]]

base.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"saved: {OUT}")
print(f"rows: {len(base):,}")
print(f"races: {base['race_id'].nunique():,}")
print("")
print("finish_status:")
print(base["finish_status"].value_counts(dropna=False))
print("")
print("ticket labels:")
print("ticket_1st :", int(base["ticket_1st"].sum()))
print("ticket_2nd :", int(base["ticket_2nd"].sum()))
print("ticket_3rd :", int(base["ticket_3rd"].sum()))
print("ticket_top3:", int(base["ticket_top3"].sum()))
print("")
print(base.head(10).to_string(index=False))
