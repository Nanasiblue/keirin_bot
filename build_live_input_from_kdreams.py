from __future__ import annotations

import argparse
import re
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd

from parse_entries_kdreams import (
    STYLE_MAP,
    find_column,
    normalize_text,
    parse_age,
    parse_name,
    pick_entry_table,
    to_float,
    to_int,
    to_rate,
)
from parse_3rentan_odds_all_fast import parse_file as parse_odds_file


OPEN_DIR = Path("data/live_input")
OUT_DIR = Path("data/live_input")

DIRECT_MODEL = Path("models/direct_ticket_lightgbm.joblib")
FINISH_MODELS = {
    "p_1st": Path("models/finish_lgbm_1st.joblib"),
    "p_top2": Path("models/finish_lgbm_top2.joblib"),
    "p_top3": Path("models/finish_lgbm_top3.joblib"),
}


def parse_place_race_no(label: str) -> tuple[str, int]:
    text = str(label or "").strip()
    m = re.search(r"(.+?競輪)\s*(\d+)R", text)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r"(\d+)R", text)
    return "", int(m.group(1)) if m else 0


def parse_entries_from_html(path: Path, race_id: str) -> pd.DataFrame:
    html = path.read_text(encoding="utf-8", errors="ignore")
    tables = pd.read_html(StringIO(html))
    df = pick_entry_table(tables)

    car_col = find_column(df, "車番")
    name_col = find_column(df, "選手名")
    age_col = find_column(df, "府県", "年齢")
    score_col = find_column(df, "競走得点")
    style_col = find_column(df, "脚質")
    win_col = find_column(df, "勝率")
    place_col = find_column(df, "3連対率")

    rows = []
    for _, row in df.iterrows():
        car_no = to_int(row[car_col])
        name = parse_name(row[name_col])
        if car_no is None or not name:
            continue

        raw_style = normalize_text(row[style_col])
        rows.append({
            "race_id": race_id,
            "car_no": car_no,
            "name": name,
            "score": to_float(row[score_col]),
            "style": STYLE_MAP.get(raw_style, raw_style),
            "age": parse_age(row[age_col]),
            "win_rate": to_rate(row[win_col]),
            "place_rate": to_rate(row[place_col]),
        })

    return pd.DataFrame(rows).sort_values(["race_id", "car_no"])


def parse_odds_from_html(path: Path) -> pd.DataFrame:
    rows, fail = parse_odds_file(str(path))
    if fail or not rows:
        return pd.DataFrame(columns=["race_id", "combination", "odds", "estimated_payout"])
    return pd.DataFrame(rows)


def add_rank_features(entries: pd.DataFrame) -> pd.DataFrame:
    out = entries.copy()
    out["score_rank_in_race"] = out.groupby("race_id")["score"].rank(ascending=False, method="min")
    out["win_rate_rank_in_race"] = out.groupby("race_id")["win_rate"].rank(ascending=False, method="min")
    out["place_rate_rank_in_race"] = out.groupby("race_id")["place_rate"].rank(ascending=False, method="min")
    return out


def race_features(entries: pd.DataFrame, open_races: pd.DataFrame) -> pd.DataFrame:
    meta = open_races.set_index("race_id").to_dict("index")
    rows = []

    for race_id, g in entries.groupby("race_id"):
        scores = pd.to_numeric(g["score"], errors="coerce")
        ages = pd.to_numeric(g["age"], errors="coerce")
        win = pd.to_numeric(g["win_rate"], errors="coerce")
        place_rate = pd.to_numeric(g["place_rate"], errors="coerce")
        style = g["style"].fillna("").astype(str)
        place, race_no = parse_place_race_no(meta.get(str(race_id), {}).get("label", ""))

        nige = int(style.str.contains("逃").sum())
        oikomi = int(style.str.contains("追").sum())
        ryo = int(style.str.contains("両").sum())
        sashi = int(style.str.contains("差").sum())
        makuri = int(style.str.contains("捲").sum())

        rows.append({
            "race_id": race_id,
            "place": place,
            "race_no": race_no,
            "grade": "",
            "weather": "unknown",
            "wind_speed": 0.0,
            "avg_score": scores.mean(),
            "max_score": scores.max(),
            "min_score": scores.min(),
            "std_score": scores.std(ddof=0),
            "score_gap": scores.max() - scores.min(),
            "avg_age": ages.mean(),
            "racer_count": len(g),
            "avg_win_rate": win.mean(),
            "max_win_rate": win.max(),
            "avg_place_rate": place_rate.mean(),
            "max_place_rate": place_rate.max(),
            "nige_count": nige,
            "oikomi_count": oikomi,
            "ryo_count": ryo,
            "sashi_count": sashi,
            "makuri_count": makuri,
            "front_runner_pressure": nige / len(g) if len(g) else 0,
            # 荒れサブ用。正式な荒れAI接続までは簡易スコア。
            "race_score": 0.5 if len(g) >= 7 else 0.0,
        })

    return pd.DataFrame(rows).fillna(0)


def make_ticket_rows(entries: pd.DataFrame, odds: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    ranked = add_rank_features(entries)
    entry_map = {
        (str(r.race_id), int(r.car_no)): r._asdict()
        for r in ranked.itertuples(index=False)
    }
    race_map = {str(r.race_id): r._asdict() for r in races.itertuples(index=False)}

    rows = []
    for o in odds.itertuples(index=False):
        race_id = str(o.race_id)
        cars = [int(x) for x in str(o.combination).split("-")]
        if len(cars) != 3:
            continue

        e1 = entry_map.get((race_id, cars[0]))
        e2 = entry_map.get((race_id, cars[1]))
        e3 = entry_map.get((race_id, cars[2]))
        rf = race_map.get(race_id, {})
        if not e1 or not e2 or not e3:
            continue

        s1, s2, s3 = float(e1["score"] or 0), float(e2["score"] or 0), float(e3["score"] or 0)
        w1, w2, w3 = float(e1["win_rate"] or 0), float(e2["win_rate"] or 0), float(e3["win_rate"] or 0)
        p1, p2, p3 = float(e1["place_rate"] or 0), float(e2["place_rate"] or 0), float(e3["place_rate"] or 0)

        row = dict(rf)
        row.update({
            "race_id": race_id,
            "combination": o.combination,
            "odds": float(o.odds),
            "estimated_payout": int(o.estimated_payout),
            "is_hit": 0,
            "hit_payout": 0,
            "hit_popularity": 0,
            "race_date": pd.to_datetime(race_id[2:10], format="%Y%m%d", errors="coerce"),
            "first_car": cars[0],
            "second_car": cars[1],
            "third_car": cars[2],
            "first_score": s1,
            "second_score": s2,
            "third_score": s3,
            "score_ticket_avg": (s1 + s2 + s3) / 3,
            "score_1_2_gap": s1 - s2,
            "score_1_3_gap": s1 - s3,
            "score_2_3_gap": s2 - s3,
            "first_age": float(e1["age"] or 0),
            "second_age": float(e2["age"] or 0),
            "third_age": float(e3["age"] or 0),
            "first_win_rate": w1,
            "second_win_rate": w2,
            "third_win_rate": w3,
            "win_rate_ticket_avg": (w1 + w2 + w3) / 3,
            "first_place_rate": p1,
            "second_place_rate": p2,
            "third_place_rate": p3,
            "place_rate_ticket_avg": (p1 + p2 + p3) / 3,
            "first_style": e1["style"],
            "second_style": e2["style"],
            "third_style": e3["style"],
            "month": pd.to_datetime(race_id[2:10], format="%Y%m%d", errors="coerce").month,
            "day": pd.to_datetime(race_id[2:10], format="%Y%m%d", errors="coerce").day,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_for_model(df: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    x = df.copy()
    for col in features:
        if col not in x.columns:
            x[col] = "unknown" if col in categorical else 0

    x = x[features].copy()
    for col in x.columns:
        if col in categorical:
            x[col] = x[col].astype("object").fillna("unknown").astype("category")
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)
    return x


def predict_direct(tickets: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(DIRECT_MODEL)
    model = bundle["model"]
    features = bundle["features"]
    categorical = bundle.get("categorical", [])

    x = prepare_for_model(tickets, features, categorical)
    out = tickets.copy()
    out["direct_ticket_score"] = model.predict_proba(x)[:, 1]
    out["direct_expected_return"] = out["direct_ticket_score"] * out["estimated_payout"]
    out["direct_ticket_rank"] = out.groupby("race_id")["direct_ticket_score"].rank(ascending=False, method="first").astype(int)
    out["direct_expected_rank"] = out.groupby("race_id")["direct_expected_return"].rank(ascending=False, method="first").astype(int)
    return out


def predict_finish(entries: pd.DataFrame) -> pd.DataFrame:
    out = add_rank_features(entries).copy()

    for pred_col, path in FINISH_MODELS.items():
        if not path.exists():
            out[pred_col] = 0.0
            continue

        bundle = joblib.load(path)
        model = bundle["model"]
        features = bundle["features"]

        x = out.copy()
        for col in features:
            if col not in x.columns:
                x[col] = 0
        x = x[features].apply(pd.to_numeric, errors="coerce").fillna(0)

        out[pred_col] = model.predict_proba(x)[:, 1]
        out[f"rank_{pred_col[2:]}"] = out.groupby("race_id")[pred_col].rank(ascending=False, method="first").astype(int)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-date", required=True)
    args = ap.parse_args()

    open_path = OPEN_DIR / f"open_races_{args.target_date}.csv"
    if not open_path.exists():
        raise SystemExit(f"open races csv not found: {open_path}")

    open_races = pd.read_csv(open_path, dtype={"race_id": str})
    print(f"open races: {len(open_races)}")

    entries_parts = []
    odds_parts = []

    for r in open_races.itertuples(index=False):
        html_path = Path(r.raw_odds_html)
        if not html_path.exists():
            print(f"[NG] missing html: {html_path}")
            continue

        try:
            entries = parse_entries_from_html(html_path, str(r.race_id))
            odds = parse_odds_from_html(html_path)
            entries_parts.append(entries)
            odds_parts.append(odds)
            print(f"[OK] {r.race_id} entries={len(entries)} odds={len(odds)}")
        except Exception as e:
            print(f"[NG] {r.race_id} {e}")

    entries_all = pd.concat(entries_parts, ignore_index=True) if entries_parts else pd.DataFrame()
    odds_all = pd.concat(odds_parts, ignore_index=True) if odds_parts else pd.DataFrame()

    if entries_all.empty or odds_all.empty:
        raise SystemExit("entries or odds is empty")

    races = race_features(entries_all, open_races)
    tickets = make_ticket_rows(entries_all, odds_all, races)
    tickets = predict_direct(tickets)

    race_score_map = races.set_index("race_id")["race_score"].to_dict()
    tickets["race_score"] = tickets["race_id"].map(race_score_map).fillna(0)

    finish = predict_finish(entries_all)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries_out = OUT_DIR / f"live_entries_{args.target_date}.csv"
    odds_out = OUT_DIR / f"live_odds_{args.target_date}.csv"
    races_out = OUT_DIR / f"live_races_{args.target_date}.csv"
    tickets_out = OUT_DIR / f"live_ticket_candidates_{args.target_date}.csv"
    finish_out = OUT_DIR / f"live_finish_predictions_{args.target_date}.csv"

    entries_all.to_csv(entries_out, index=False, encoding="utf-8-sig")
    odds_all.to_csv(odds_out, index=False, encoding="utf-8-sig")
    races.to_csv(races_out, index=False, encoding="utf-8-sig")
    tickets.to_csv(tickets_out, index=False, encoding="utf-8-sig")
    finish.to_csv(finish_out, index=False, encoding="utf-8-sig")

    print("")
    print(f"saved: {entries_out}")
    print(f"saved: {odds_out}")
    print(f"saved: {races_out}")
    print(f"saved: {tickets_out}")
    print(f"saved: {finish_out}")
    print("")
    print(f"entries rows: {len(entries_all):,}")
    print(f"odds rows: {len(odds_all):,}")
    print(f"ticket rows: {len(tickets):,}")
    print(f"finish rows: {len(finish):,}")


if __name__ == "__main__":
    main()
