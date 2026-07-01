from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

MODEL = Path("models/direct_ticket_lightgbm.joblib")
ODDS = Path("data/odds_3rentan_all.csv")
ENTRIES = Path("data/entries_all_kdreams.csv")
PAYOUTS = Path("data/payouts_all_kdreams.csv")
FEATURES = Path("data/features_rich.csv")

PRED_OUT = Path("data/direct_ticket_predictions_full_valid.csv")
SIM_OUT = Path("data/direct_ticket_full_valid_simulation.csv")

START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-06-26")


def race_date_from_id(race_id):
    race_id = race_id.astype(str)
    start = pd.to_datetime(race_id.str[2:10], format="%Y%m%d", errors="coerce")
    day_no = pd.to_numeric(race_id.str[10:12], errors="coerce").fillna(1).astype(int)
    return start + pd.to_timedelta(day_no - 1, unit="D")


def split_combination(df):
    cars = df["combination"].astype(str).str.split("-", expand=True)
    df["first_car"] = pd.to_numeric(cars[0], errors="coerce").astype("Int64")
    df["second_car"] = pd.to_numeric(cars[1], errors="coerce").astype("Int64")
    df["third_car"] = pd.to_numeric(cars[2], errors="coerce").astype("Int64")
    return df


def load_entries():
    e = pd.read_csv(ENTRIES, dtype={"race_id": str})
    e["car_no"] = pd.to_numeric(e["car_no"], errors="coerce").astype("Int64")
    for col in ["score", "age", "win_rate", "place_rate"]:
        e[col] = pd.to_numeric(e[col], errors="coerce")
    return e[["race_id", "car_no", "score", "style", "age", "win_rate", "place_rate"]].copy()


def add_car_features(df, entries):
    for pos, car_col in [("first", "first_car"), ("second", "second_car"), ("third", "third_car")]:
        tmp = entries.rename(columns={
            "car_no": car_col,
            "score": f"{pos}_score",
            "style": f"{pos}_style",
            "age": f"{pos}_age",
            "win_rate": f"{pos}_win_rate",
            "place_rate": f"{pos}_place_rate",
        })
        df = df.merge(tmp, on=["race_id", car_col], how="left")
    return df


def add_ticket_features(df):
    df["score_1_2_gap"] = df["first_score"] - df["second_score"]
    df["score_1_3_gap"] = df["first_score"] - df["third_score"]
    df["score_2_3_gap"] = df["second_score"] - df["third_score"]
    df["score_ticket_avg"] = df[["first_score", "second_score", "third_score"]].mean(axis=1)
    df["win_rate_ticket_avg"] = df[["first_win_rate", "second_win_rate", "third_win_rate"]].mean(axis=1)
    df["place_rate_ticket_avg"] = df[["first_place_rate", "second_place_rate", "third_place_rate"]].mean(axis=1)
    return df


def load_hit_map():
    p = pd.read_csv(PAYOUTS, dtype={"race_id": str, "combination": str})
    p = p[p["bet_type"].astype(str).eq("3連単")].copy()
    p = p[["race_id", "combination", "payout", "popularity"]].rename(columns={
        "combination": "hit_combination",
        "payout": "hit_payout",
        "popularity": "hit_popularity",
    })
    p["hit_payout"] = pd.to_numeric(p["hit_payout"], errors="coerce").fillna(0)
    return p


def load_race_features():
    f = pd.read_csv(FEATURES, dtype={"race_id": str})
    keep = [
        "race_id", "avg_score", "max_score", "min_score", "std_score", "score_gap",
        "avg_age", "racer_count", "avg_win_rate", "max_win_rate",
        "avg_place_rate", "max_place_rate", "nige_count", "oikomi_count",
        "ryo_count", "sashi_count", "makuri_count", "front_runner_pressure",
        "place", "race_no", "weather", "wind_speed", "is_rain", "is_strong_wind",
    ]
    keep = [c for c in keep if c in f.columns]
    return f[keep].copy()


def make_model_matrix(df, feature_cols, categorical_cols):
    X = pd.DataFrame(index=df.index)

    for col in feature_cols:
        X[col] = df[col] if col in df.columns else 0

    for col in feature_cols:
        if col in categorical_cols:
            X[col] = X[col].astype("object").fillna("unknown").astype("category")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X


def top_per_race(df, metric, n):
    return (
        df.sort_values(["race_id", metric], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(n)
    )


def simulate(selected, rule):
    races = selected["race_id"].nunique()
    tickets = len(selected)
    bet = tickets * 100
    ret = int(selected["return_yen"].sum()) if tickets else 0
    hit = int(selected["is_hit"].sum()) if tickets else 0
    max_hit = int(selected["return_yen"].max()) if tickets else 0

    return {
        "rule": rule,
        "race_count": races,
        "tickets": tickets,
        "avg_tickets_per_race": tickets / races if races else 0,
        "bet": bet,
        "return": ret,
        "profit": ret - bet,
        "roi": ret / bet if bet else 0,
        "roi_without_max_hit": (ret - max_hit) / bet if bet else 0,
        "hit_count": hit,
        "hit_rate_per_ticket": hit / tickets if tickets else 0,
        "hit_rate_per_race": hit / races if races else 0,
        "avg_score": selected["direct_ticket_score"].mean() if tickets else 0,
        "min_score": selected["direct_ticket_score"].min() if tickets else 0,
        "avg_odds": selected["odds"].mean() if tickets else 0,
        "median_odds": selected["odds"].median() if tickets else 0,
        "avg_direct_expected_return": selected["direct_expected_return"].mean() if tickets else 0,
        "avg_hit_payout": selected.loc[selected["is_hit"].eq(1), "return_yen"].mean() if hit else 0,
        "max_hit_payout": max_hit,
        "max_hit_share": max_hit / ret if ret else 0,
    }


def score_full_valid(chunksize=1_000_000):
    bundle = joblib.load(MODEL)
    model = bundle["model"]
    feature_cols = bundle["features"]
    categorical_cols = bundle.get("categorical", [])

    entries = load_entries()
    hit = load_hit_map()
    race_features = load_race_features()

    outputs = []
    scanned = 0
    kept = 0
    hit_total = 0

    for i, chunk in enumerate(pd.read_csv(ODDS, dtype={"race_id": str, "combination": str}, chunksize=chunksize), start=1):
        scanned += len(chunk)
        chunk["race_date"] = race_date_from_id(chunk["race_id"])
        chunk = chunk[(chunk["race_date"] >= START) & (chunk["race_date"] <= END)].copy()

        if chunk.empty:
            print(f"chunk {i}: scanned={scanned:,} valid_rows={kept:,}")
            continue

        chunk["odds"] = pd.to_numeric(chunk["odds"], errors="coerce").fillna(0)
        chunk["estimated_payout"] = pd.to_numeric(chunk["estimated_payout"], errors="coerce").fillna(0)

        chunk = split_combination(chunk)
        chunk = add_car_features(chunk, entries)
        chunk = add_ticket_features(chunk)
        chunk = chunk.merge(race_features, on="race_id", how="left")
        chunk["year"] = chunk["race_date"].dt.year
        chunk["month"] = chunk["race_date"].dt.month

        chunk = chunk.merge(hit, on="race_id", how="left")
        chunk["is_hit"] = (chunk["combination"] == chunk["hit_combination"]).astype(int)
        chunk["return_yen"] = chunk["is_hit"] * chunk["hit_payout"].fillna(0)

        X = make_model_matrix(chunk, feature_cols, categorical_cols)
        pred = model.predict_proba(X)[:, 1]

        out = chunk[[
            "race_id", "combination", "race_date", "odds", "estimated_payout",
            "is_hit", "hit_payout", "hit_popularity", "return_yen",
        ]].copy()

        out["direct_ticket_score"] = pred
        out["direct_expected_return"] = out["direct_ticket_score"] * out["estimated_payout"]

        outputs.append(out)
        kept += len(out)
        hit_total += int(out["is_hit"].sum())

        print(f"chunk {i}: scanned={scanned:,} valid_rows={kept:,} hits={hit_total:,}")

    df = pd.concat(outputs, ignore_index=True)

    df["direct_ticket_rank"] = df.groupby("race_id")["direct_ticket_score"].rank(
        ascending=False,
        method="first",
    ).astype(int)

    df["direct_expected_rank"] = df.groupby("race_id")["direct_expected_return"].rank(
        ascending=False,
        method="first",
    ).astype(int)

    df.to_csv(PRED_OUT, index=False, encoding="utf-8-sig")

    print(f"saved: {PRED_OUT}")
    print(f"full valid rows: {len(df):,}")
    print(f"races: {df['race_id'].nunique():,}")
    print(f"hits: {int(df['is_hit'].sum()):,}")

    return df


def run_simulation(df):
    rows = []

    odds_ranges = [
        ("odds_all", 0, 999999),
        ("odds_10plus", 10, 999999),
        ("odds_30plus", 30, 999999),
        ("odds_50plus", 50, 999999),
        ("odds_100plus", 100, 999999),
        ("odds_30_500", 30, 500),
        ("odds_50_500", 50, 500),
        ("odds_50_1000", 50, 1000),
        ("odds_100_1000", 100, 1000),
    ]

    top_ns = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
    score_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    expected_thresholds = [50, 80, 100, 150, 200, 300, 500, 800, 1000]

    for odds_name, lo, hi in odds_ranges:
        base = df[(df["odds"] >= lo) & (df["odds"] <= hi)].copy()
        if base.empty:
            continue

        for metric in ["direct_ticket_score", "direct_expected_return"]:
            for n in top_ns:
                rows.append(simulate(
                    top_per_race(base, metric, n),
                    f"{odds_name}_top{n}_by_{metric}",
                ))

        for th in score_thresholds:
            b = base[base["direct_ticket_score"] >= th].copy()
            if b.empty:
                continue

            rows.append(simulate(b, f"{odds_name}_score>={th}"))

            for n in [1, 2, 3, 5, 10]:
                rows.append(simulate(
                    top_per_race(b, "direct_ticket_score", n),
                    f"{odds_name}_score>={th}_top{n}",
                ))

        for th in expected_thresholds:
            b = base[base["direct_expected_return"] >= th].copy()
            if b.empty:
                continue

            rows.append(simulate(b, f"{odds_name}_expected>={th}"))

            for n in [1, 2, 3, 5, 10]:
                rows.append(simulate(
                    top_per_race(b, "direct_expected_return", n),
                    f"{odds_name}_expected>={th}_top{n}",
                ))

    sim = pd.DataFrame(rows).sort_values(
        ["roi_without_max_hit", "roi", "hit_count"],
        ascending=[False, False, False],
    )

    sim.to_csv(SIM_OUT, index=False, encoding="utf-8-sig")
    print(f"saved: {SIM_OUT}")

    print(sim.head(80).to_string(
        index=False,
        formatters={
            "roi": lambda x: f"{x:.2%}",
            "roi_without_max_hit": lambda x: f"{x:.2%}",
            "hit_rate_per_ticket": lambda x: f"{x:.2%}",
            "hit_rate_per_race": lambda x: f"{x:.2%}",
            "max_hit_share": lambda x: f"{x:.2%}",
            "avg_tickets_per_race": lambda x: f"{x:.1f}",
            "avg_score": lambda x: f"{x:.4f}",
            "min_score": lambda x: f"{x:.4f}",
            "avg_odds": lambda x: f"{x:.1f}",
            "median_odds": lambda x: f"{x:.1f}",
            "avg_direct_expected_return": lambda x: f"{x:.1f}",
            "avg_hit_payout": lambda x: f"{x:.0f}",
            "max_hit_payout": lambda x: f"{x:.0f}",
        },
    ))


def main():
    df = score_full_valid()
    run_simulation(df)


if __name__ == "__main__":
    main()
