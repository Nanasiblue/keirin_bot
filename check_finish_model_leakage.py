from pathlib import Path
import re
import pandas as pd

DATA = Path("data")
IN = DATA / "racer_rank_dataset.csv"

DROP_COLS = {
    "race_id", "race_date", "name",
    "finish_position", "finish_class",
    "margin", "agari", "decision", "sb", "comment",
    "combination", "payout", "popularity",
    "ticket_1st", "ticket_2nd", "ticket_3rd", "ticket_top2", "ticket_top3",
    "official_is_1st", "official_top2", "official_top3",
    "ticket_1st_car", "ticket_2nd_car", "ticket_3rd_car",
    "has_numeric_finish", "is_finished",
}

NG_PATTERNS = [
    "finish", "ticket", "official", "payout", "popularity", "combination",
    "result", "rank", "着", "順位", "払戻", "決まり", "decision",
    "margin", "agari", "comment",
]

df = pd.read_csv(IN, dtype={"race_id": str}, low_memory=False)
df["race_id"] = df["race_id"].astype(str).str.zfill(16)
df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

train = df[df["race_date"] <= "2025-12-31"].copy()
valid = df[(df["race_date"] >= "2026-01-01") & (df["race_date"] <= "2026-06-26")].copy()

feature_cols = [c for c in df.columns if c not in DROP_COLS]
suspect_cols = []
for c in feature_cols:
    low = c.lower()
    if any(p.lower() in low for p in NG_PATTERNS):
        suspect_cols.append(c)

train_races = set(train["race_id"].unique())
valid_races = set(valid["race_id"].unique())
overlap = train_races & valid_races

print("=== split ===")
print(f"all rows   : {len(df):,}")
print(f"train rows : {len(train):,}")
print(f"valid rows : {len(valid):,}")
print(f"train races: {len(train_races):,}")
print(f"valid races: {len(valid_races):,}")
print(f"overlap races: {len(overlap):,}")

print("\n=== target distribution ===")
print("train:")
print(train["finish_class"].value_counts(dropna=False).to_string())
print("\nvalid:")
print(valid["finish_class"].value_counts(dropna=False).to_string())

print("\n=== feature columns ===")
print(f"feature count: {len(feature_cols):,}")

print("\n=== suspect feature columns ===")
if suspect_cols:
    print(f"suspect count: {len(suspect_cols):,}")
    for c in suspect_cols[:200]:
        print(c)
else:
    print("なし")

print("\n=== columns explicitly dropped but existing ===")
existing_drop = [c for c in DROP_COLS if c in df.columns]
for c in sorted(existing_drop):
    print(c)

print("\n=== verdict ===")
if overlap:
    print("NG: train/validで同じrace_idが混ざっています")
elif suspect_cols:
    print("要確認: 怪しい特徴量名があります。上の列が本当に未来情報でないか確認してください")
else:
    print("OK: 明らかなリーク列は見当たりません")
