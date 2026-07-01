from pathlib import Path
import pandas as pd

DATA = Path("data")
IN_RESULTS = DATA / "race_results_full_clean_v2.csv"
IN_FEATURES = DATA / "features_rich.csv"
OUT = DATA / "racer_rank_dataset.csv"

def add_race_date(df):
    rid = df["race_id"].astype(str).str.zfill(16)
    start = pd.to_datetime(rid.str[2:10], format="%Y%m%d", errors="coerce")
    day_no = pd.to_numeric(rid.str[10:12], errors="coerce").fillna(1).astype(int)
    df["race_date"] = start + pd.to_timedelta(day_no - 1, unit="D")
    return df

r = pd.read_csv(IN_RESULTS, dtype={"race_id": str})
f = pd.read_csv(IN_FEATURES, dtype={"race_id": str})

r["race_id"] = r["race_id"].astype(str).str.zfill(16)
f["race_id"] = f["race_id"].astype(str).str.zfill(16)

r = add_race_date(r)

# 三連単の正解に直結する分類を優先
r["finish_class"] = "4plus"
r.loc[r["ticket_1st"] == 1, "finish_class"] = "1st"
r.loc[r["ticket_2nd"] == 1, "finish_class"] = "2nd"
r.loc[r["ticket_3rd"] == 1, "finish_class"] = "3rd"

exception_status = ["crash", "crash_dnf", "dnf", "disqualified", "accident_finish", "unknown_exception"]
r.loc[(r["finish_status_detail"].isin(exception_status)) & (r["ticket_top3"] == 0), "finish_class"] = "exception"
r.loc[r["finish_status_detail"].eq("absent"), "finish_class"] = "absent"

# レース内相対特徴
r["score_rank_in_race"] = r.groupby("race_id")["score"].rank(ascending=False, method="min")
r["win_rate_rank_in_race"] = r.groupby("race_id")["win_rate"].rank(ascending=False, method="min")
r["place_rate_rank_in_race"] = r.groupby("race_id")["place_rate"].rank(ascending=False, method="min")
r["score_diff_from_race_avg"] = r["score"] - r.groupby("race_id")["score"].transform("mean")
r["score_diff_from_race_max"] = r["score"] - r.groupby("race_id")["score"].transform("max")

drop_feature_cols = [
    "combination", "payout_3rentan", "payout_popularity",
    "is_over_30", "is_over_50", "is_over_100",
    "is_popular_50plus", "is_popular_100plus", "is_popular_150plus",
]
race_features = f.drop(columns=[c for c in drop_feature_cols if c in f.columns], errors="ignore")

df = r.merge(race_features, on="race_id", how="left", suffixes=("", "_race"))

df.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"saved: {OUT}")
print(f"rows: {len(df):,}")
print(f"races: {df['race_id'].nunique():,}")
print("")
print(df["finish_class"].value_counts(dropna=False).to_string())
print("")
print(df.head(10).to_string(index=False))
