import pandas as pd

df = pd.read_csv("data/ticket_value_simulation.csv")

df["roi_without_max_hit"] = (df["return"] - df["max_hit_payout"]) / df["bet"]
df["max_hit_share"] = df["max_hit_payout"] / df["return"].replace(0, pd.NA)

stable = df[
    (df["race_count"] >= 300)
    & (df["tickets"] >= 1000)
    & (df["hit_count"] >= 20)
].copy()

stable = stable.sort_values(
    ["roi_without_max_hit", "roi", "hit_count"],
    ascending=[False, False, False]
)

print("=== stable rules: hit_count>=20 ===")
print(stable.head(40).to_string(
    index=False,
    formatters={
        "roi": lambda x: f"{x:.2%}",
        "roi_without_max_hit": lambda x: f"{x:.2%}",
        "max_hit_share": lambda x: f"{x:.2%}" if pd.notna(x) else "0.00%",
    }
))

stable2 = df[
    (df["race_count"] >= 300)
    & (df["tickets"] >= 3000)
    & (df["hit_count"] >= 50)
].copy()

stable2 = stable2.sort_values(
    ["roi_without_max_hit", "roi", "hit_count"],
    ascending=[False, False, False]
)

print("\n=== more stable rules: hit_count>=50 ===")
print(stable2.head(40).to_string(
    index=False,
    formatters={
        "roi": lambda x: f"{x:.2%}",
        "roi_without_max_hit": lambda x: f"{x:.2%}",
        "max_hit_share": lambda x: f"{x:.2%}" if pd.notna(x) else "0.00%",
    }
))
