from pathlib import Path
import pandas as pd

DATA = Path("data")

results = pd.read_csv(DATA / "race_results_full.csv", dtype={"race_id": str})
entries = pd.read_csv(DATA / "entries_all_kdreams.csv", dtype={"race_id": str})
payouts = pd.read_csv(DATA / "payouts_all_kdreams.csv", dtype={"race_id": str})

for df in [results, entries, payouts]:
    df["race_id"] = df["race_id"].astype(str).str.zfill(16)

results["finish_position"] = pd.to_numeric(results["finish_position"], errors="coerce")
results["car_no"] = pd.to_numeric(results["car_no"], errors="coerce")
entries["car_no"] = pd.to_numeric(entries["car_no"], errors="coerce")

print("=== 基本件数 ===")
print(f"results rows : {len(results):,}")
print(f"results races: {results['race_id'].nunique():,}")
print(f"entries rows : {len(entries):,}")
print(f"entries races: {entries['race_id'].nunique():,}")
print(f"payout races : {payouts['race_id'].nunique():,}")

print("\n=== 欠損 ===")
print(results[["race_id", "finish_position", "car_no", "name"]].isna().sum())

print("\n=== 重複 ===")
print("race_id+car_no:", results.duplicated(["race_id", "car_no"]).sum())
print("race_id+finish_position:", results.duplicated(["race_id", "finish_position"]).sum())

print("\n=== 出走人数分布 ===")
result_counts = results.groupby("race_id").size()
print(result_counts.value_counts().sort_index())

print("\n=== entries人数との差 ===")
entry_counts = entries.groupby("race_id").size()
count_check = pd.concat(
    [entry_counts.rename("entry_count"), result_counts.rename("result_count")],
    axis=1
).fillna(0).astype(int)
count_check["diff"] = count_check["result_count"] - count_check["entry_count"]
diff_rows = count_check[count_check["diff"] != 0].reset_index()
print(f"人数差あり: {len(diff_rows):,}")
if len(diff_rows):
    diff_rows.to_csv(DATA / "race_results_entry_count_diff.csv", index=False, encoding="utf-8-sig")
    print(diff_rows.head(20).to_string(index=False))

print("\n=== 三連単1-3着との一致 ===")
top3 = results[results["finish_position"].isin([1, 2, 3])].copy()
top3 = top3.sort_values(["race_id", "finish_position"])
actual = top3.groupby("race_id")["car_no"].apply(
    lambda x: "-".join(str(int(v)) for v in x)
).reset_index(name="result_top3")

pay3 = payouts[payouts["bet_type"].astype(str).str.contains("3連単", na=False)].copy()
pay3 = pay3[["race_id", "combination", "payout", "popularity"]].drop_duplicates("race_id")

merged = pay3.merge(actual, on="race_id", how="left")
merged["top3_match"] = merged["combination"].astype(str) == merged["result_top3"].astype(str)

total = len(merged)
matched = int(merged["top3_match"].sum())
print(f"比較対象: {total:,}")
print(f"一致: {matched:,} / {total:,} ({matched / total:.2%})")
print(f"結果top3欠損: {merged['result_top3'].isna().sum():,}")

mismatch = merged[~merged["top3_match"]].copy()
print(f"不一致: {len(mismatch):,}")
if len(mismatch):
    mismatch.to_csv(DATA / "race_results_top3_mismatch.csv", index=False, encoding="utf-8-sig")
    print(mismatch.head(20).to_string(index=False))

print("\n=== 判定 ===")
if total and matched / total >= 0.995 and len(diff_rows) < 1000:
    print("OK: 順位予測用データとして使えそうです。")
else:
    print("要確認: 不一致または人数差が多いです。")
