from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

def summarize(base, rule, selected):
    n = len(selected)
    return {
        "rule": rule,
        "selected_races": n,
        "selected_rate": n / len(base) if len(base) else 0,
        "avg_score": selected["ai_score"].mean() if n else 0,
        "min_score": selected["ai_score"].min() if n else 0,
        "avg_payout": selected["payout_3rentan"].mean() if n else 0,
        "median_payout": selected["payout_3rentan"].median() if n else 0,
        "max_payout": selected["payout_3rentan"].max() if n else 0,
        "over_30_rate": (selected["payout_3rentan"] >= 30000).mean() if n else 0,
        "over_50_rate": (selected["payout_3rentan"] >= 50000).mean() if n else 0,
        "over_100_rate": (selected["payout_3rentan"] >= 100000).mean() if n else 0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/predictions_valid_2026_is_over_50.csv")
    parser.add_argument("--output", default="data/prediction_threshold_summary_is_over_50.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["ai_score"] = pd.to_numeric(df["ai_score"], errors="coerce")
    df["payout_3rentan"] = pd.to_numeric(df["payout_3rentan"], errors="coerce")
    df = df.dropna(subset=["ai_score", "payout_3rentan"]).copy()
    df = df.sort_values("ai_score", ascending=False)

    rows = [summarize(df, "ALL", df)]

    for top_n in [10, 20, 30, 50, 100, 200, 300, 500]:
        rows.append(summarize(df, f"top_{top_n}", df.head(top_n)))

    for pct in [0.01, 0.03, 0.05, 0.10]:
        rows.append(summarize(df, f"top_{int(pct * 100)}pct", df.head(max(1, int(len(df) * pct)))))

    for th in [0.80, 0.78, 0.75, 0.72, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
        rows.append(summarize(df, f"score>={th:.2f}", df[df["ai_score"] >= th]))

    result = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"saved: {args.output}")
    print(result.to_string(index=False, formatters={
        "selected_rate": "{:.2%}".format,
        "avg_score": "{:.4f}".format,
        "min_score": "{:.4f}".format,
        "avg_payout": "{:.0f}".format,
        "median_payout": "{:.0f}".format,
        "max_payout": "{:.0f}".format,
        "over_30_rate": "{:.2%}".format,
        "over_50_rate": "{:.2%}".format,
        "over_100_rate": "{:.2%}".format,
    }))

if __name__ == "__main__":
    main()
