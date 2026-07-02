from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd


LIVE_DIR = Path("data/live_predictions")
LOG_DIR = Path("data/live_logs")

HIST_TICKETS = Path("data/direct_ticket_predictions_full_valid.csv")
HIST_FINISH = Path("data/finish_lightgbm_predictions_valid.csv")
CONTEXT_FILES = [
    Path("data/features_rich.csv"),
    Path("data/features_all_kdreams.csv"),
    Path("data/features_from_kdreams.csv"),
]

MAIN_RULE = "direct_v1_middle_odds"
HIGH_RULE = "high_odds_v1_top5"
WATCH_RULE = "huge_odds_watch"

STABLE_RULE = {
    "min_odds": 10,
    "max_odds": 100,
    "min_ticket_score": 0.65,
    "max_per_race": 40,
    "stake_yen": 100,
    "label": "安定",
    "reason": "安定: 低〜中オッズ + スコア候補",
}

HIGH_ODDS_RULE = {
    "min_odds": 80,
    "max_odds": 800,
    "min_ticket_score": 0.30,
    "min_race_score": 0.25,
    "min_expected_return": 2000,
    "max_per_race": 40,
    "stake_yen": 100,
    "label": "荒れ",
    "reason": "荒れ: 高オッズ + 荒れスコア候補",
}

WATCH_ODDS_RULE = {
    "min_odds": 200,
    "max_odds": 3000,
    "min_ticket_score": 0.10,
    "min_race_score": 0.25,
    "min_expected_return": 0,
    "max_per_race": 40,
    "stake_yen": 0,
    "label": "大荒れ見るだけ",
    "reason": "大荒れウォッチ: 200倍以上 + 荒れ気配",
}

def today_jst() -> str:
    return (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d")


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False, **kwargs)


def detect_score_col(df: pd.DataFrame) -> str | None:
    for col in ["ai_score", "race_score", "pred_score", "score", "prob", "proba"]:
        if col in df.columns:
            return col
    return None


def load_race_context() -> pd.DataFrame:
    parts = []


    for path in CONTEXT_FILES:
        if not path.exists():
            continue
        ctx = read_csv_if_exists(path)
        keep = [c for c in ["race_id", "place", "race_no", "deadline_jst", "grade", "weather", "wind_speed"] if c in ctx.columns]
        if "race_id" in keep and len(keep) > 1:
            parts.append(ctx[keep].drop_duplicates("race_id"))
            print(f"context loaded: {path} / columns={keep}")

    if not parts:
        return pd.DataFrame(columns=["race_id"])

    out = parts[0].drop_duplicates("race_id").copy()
    for p in parts[1:]:
        out = out.merge(p.drop_duplicates("race_id"), on="race_id", how="outer", suffixes=("", "_ctx"))
        for col in ["place", "race_no", "deadline_jst", "grade", "weather", "wind_speed"]:
            alt = f"{col}_ctx"
            if alt in out.columns:
                if col not in out.columns:
                    out[col] = out[alt]
                else:
                    out[col] = out[col].where(out[col].notna() & (out[col].astype(str) != ""), out[alt])
                out = out.drop(columns=[alt])

    return out.drop_duplicates("race_id")


def load_ticket_candidates(target_date: str) -> pd.DataFrame:
    live_file = Path(f"data/live_input/live_ticket_candidates_{target_date}.csv")
    if live_file.exists():
        print(f"loading live candidates: {live_file}")
        return pd.read_csv(live_file, dtype={"race_id": str, "combination": str})

    if HIST_TICKETS.exists():
        print(f"loading historical ticket predictions for dry run: {HIST_TICKETS}")
        df = pd.read_csv(HIST_TICKETS, dtype={"race_id": str, "combination": str}, low_memory=False)
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df[df["race_date"] == target_date].copy()
        if not df.empty:
            return df

    raise SystemExit(
        f"live候補CSVがありません: {live_file}\n"
        "まずは 2026-01-01 など過去検証にある日付で試してください。"
    )


def load_finish_predictions(target_date: str) -> pd.DataFrame:
    live_file = Path(f"data/live_input/live_finish_predictions_{target_date}.csv")
    if live_file.exists():
        print(f"loading live finish predictions: {live_file}")
        return pd.read_csv(live_file, dtype={"race_id": str})

    if HIST_FINISH.exists():
        print(f"loading historical finish predictions for dry run: {HIST_FINISH}")
        df = pd.read_csv(HIST_FINISH, dtype={"race_id": str}, low_memory=False)
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return df[df["race_date"] == target_date].copy()

    return pd.DataFrame()


def enrich_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ctx = load_race_context()
    if ctx.empty:
        print("WARNING: race context not found. place/race_score may be empty.")
        return df

    before = len(df)
    df = df.merge(ctx, on="race_id", how="left", suffixes=("", "_ctx"))

    for col in ["place", "race_no", "deadline_jst", "grade", "weather", "wind_speed"]:
        alt = f"{col}_ctx"
        if alt in df.columns:
            if col not in df.columns:
                df[col] = df[alt]
            else:
                df[col] = df[col].where(df[col].notna() & (df[col].astype(str) != ""), df[alt])
            df = df.drop(columns=[alt])

    print(f"candidate context merged: {before:,} rows")
    return df


def normalize_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "race_date" in df.columns:
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for col in ["odds", "direct_ticket_score", "race_score", "direct_expected_return", "estimated_payout"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "race_no" not in df.columns:
        df["race_no"] = pd.to_numeric(df["race_id"].astype(str).str[-2:], errors="coerce").fillna(0).astype(int)
    else:
        df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)

    if "place" not in df.columns:
        df["place"] = ""
    df["place"] = df["place"].fillna("").astype(str)

    if "direct_expected_return" not in df.columns or (df["direct_expected_return"] == 0).all():
        df["direct_expected_return"] = df["direct_ticket_score"] * df["odds"] * 100

    missing_place = (df["place"] == "").sum()
    zero_race_score = (df["race_score"] == 0).sum()
    print(f"place missing rows: {missing_place:,} / {len(df):,}")
    print(f"race_score zero rows: {zero_race_score:,} / {len(df):,}")

    return df


def select_main(df: pd.DataFrame) -> pd.DataFrame:
    base = df[
        (df["odds"] >= STABLE_RULE["min_odds"])
        & (df["odds"] <= STABLE_RULE["max_odds"])
        & (df["direct_ticket_score"] >= STABLE_RULE["min_ticket_score"])
    ].copy()

    if base.empty:
        return base

    selected = (
        base.sort_values(["race_id", "direct_ticket_score"], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(STABLE_RULE["max_per_race"])
        .copy()
    )
    selected["rule"] = MAIN_RULE
    selected["stake_yen"] = STABLE_RULE["stake_yen"]
    selected["strategy_label"] = STABLE_RULE["label"]
    selected["reason"] = STABLE_RULE["reason"]
    return selected

def select_high(df: pd.DataFrame) -> pd.DataFrame:
    base = df[
        (df["odds"] >= HIGH_ODDS_RULE["min_odds"])
        & (df["odds"] <= HIGH_ODDS_RULE["max_odds"])
        & (df["race_score"] >= HIGH_ODDS_RULE["min_race_score"])
        & (df["direct_ticket_score"] >= HIGH_ODDS_RULE["min_ticket_score"])
        & (df["direct_expected_return"] >= HIGH_ODDS_RULE["min_expected_return"])
    ].copy()

    if base.empty:
        return base

    selected = (
        base.sort_values(["race_id", "direct_expected_return"], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(HIGH_ODDS_RULE["max_per_race"])
        .copy()
    )
    selected["rule"] = HIGH_RULE
    selected["stake_yen"] = HIGH_ODDS_RULE["stake_yen"]
    selected["strategy_label"] = HIGH_ODDS_RULE["label"]
    selected["reason"] = HIGH_ODDS_RULE["reason"]
    return selected


def select_watch(df: pd.DataFrame) -> pd.DataFrame:
    base = df[
        (df["odds"] >= WATCH_ODDS_RULE["min_odds"])
        & (df["odds"] <= WATCH_ODDS_RULE["max_odds"])
        & (df["race_score"] >= WATCH_ODDS_RULE["min_race_score"])
        & (df["direct_ticket_score"] >= WATCH_ODDS_RULE["min_ticket_score"])
        & (df["direct_expected_return"] >= WATCH_ODDS_RULE["min_expected_return"])
    ].copy()

    if base.empty:
        return base

    selected = (
        base.sort_values(["race_id", "direct_expected_return"], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(WATCH_ODDS_RULE["max_per_race"])
        .copy()
    )
    selected["rule"] = WATCH_RULE
    selected["stake_yen"] = WATCH_ODDS_RULE["stake_yen"]
    selected["strategy_label"] = WATCH_ODDS_RULE["label"]
    selected["reason"] = WATCH_ODDS_RULE["reason"]
    return selected

def make_finish_summary(finish: pd.DataFrame) -> dict[str, str]:
    if finish.empty:
        return {}

    out = {}
    for race_id, g in finish.groupby("race_id"):
        lines = []

        if "p_1st" in g.columns:
            top = g.sort_values("p_1st", ascending=False).head(3)
            vals = [f"{int(r.car_no)}番 {r.p_1st:.2f}" for r in top.itertuples()]
            lines.append("1着候補: " + ", ".join(vals))

        if "p_top2" in g.columns:
            top = g.sort_values("p_top2", ascending=False).head(3)
            vals = [f"{int(r.car_no)}番 {r.p_top2:.2f}" for r in top.itertuples()]
            lines.append("2着以内: " + ", ".join(vals))

        if "p_top3" in g.columns:
            top = g.sort_values("p_top3", ascending=False).head(4)
            vals = [f"{int(r.car_no)}番 {r.p_top3:.2f}" for r in top.itertuples()]
            lines.append("3着以内: " + ", ".join(vals))

        out[str(race_id)] = "\n".join(lines)

    return out


def title_for_race(first: pd.Series, race_id: str) -> str:
    place = str(first.get("place", "")).strip()
    race_no = int(first.get("race_no", 0))
    if place:
        return f"{place} {race_no}R"
    return f"{race_id} {race_no}R"


def deadline_for_race(first: pd.Series) -> str:
    raw = str(first.get("deadline_jst", "")).strip()
    if not raw or raw.lower() == "nan":
        return ""
    dt_value = pd.to_datetime(raw, errors="coerce")
    if pd.isna(dt_value):
        return raw
    return dt_value.strftime("%H:%M")


def short_rule_name(rule: str) -> str:
    if rule == MAIN_RULE:
        return STABLE_RULE["label"]
    if rule == HIGH_RULE:
        return HIGH_ODDS_RULE["label"]
    if rule == WATCH_RULE:
        return WATCH_ODDS_RULE["label"]
    return str(rule)
def simplify_finish_line(line: str) -> str:
    line = line.replace("1着候補: ", "1着 ")
    line = line.replace("2着以内: ", "2内 ")
    line = line.replace("3着以内: ", "3内 ")
    line = line.replace("番 ", "(").replace(", ", ") ")
    if "(" in line and not line.endswith(")"):
        line += ")"
    return line


def make_summary_text(target_date: str, tickets: pd.DataFrame, finish_map: dict[str, str]) -> str:
    total_tickets = len(tickets)
    total_races = tickets["race_id"].nunique() if not tickets.empty else 0
    total_stake = int(tickets["stake_yen"].sum()) if not tickets.empty else 0

    lines = [
        f"競輪AI {target_date}",
        f"候補一覧: {total_tickets}点 / {total_races}R / 全点表示",
        (
            f"安定: {STABLE_RULE['min_odds']}-{STABLE_RULE['max_odds']}倍"
            f" score>={STABLE_RULE['min_ticket_score']:.2f}"
            f" / 各R最大{STABLE_RULE['max_per_race']}点"
        ),
        (
            f"荒れ: {HIGH_ODDS_RULE['min_odds']}-{HIGH_ODDS_RULE['max_odds']}倍"
            f" score>={HIGH_ODDS_RULE['min_ticket_score']:.2f}"
            f" race>={HIGH_ODDS_RULE['min_race_score']:.2f}"
            f" / 各R最大{HIGH_ODDS_RULE['max_per_race']}点"
        ),
        (
            f"大荒れ見るだけ: {WATCH_ODDS_RULE['min_odds']}-{WATCH_ODDS_RULE['max_odds']}倍"
            f" score>={WATCH_ODDS_RULE['min_ticket_score']:.2f}"
            f" race>={WATCH_ODDS_RULE['min_race_score']:.2f}"
            f" / 各R最大{WATCH_ODDS_RULE['max_per_race']}点"
        ),
        "",
    ]

    if tickets.empty:
        lines.append("今日は買い目候補なし")
        return "\n".join(lines)

    rule_order = {MAIN_RULE: 0, HIGH_RULE: 1, WATCH_RULE: 2}
    tickets = tickets.copy()
    tickets["_rule_order"] = tickets["rule"].map(rule_order).fillna(9)
    tickets["_deadline_sort"] = pd.to_datetime(tickets["deadline_jst"], errors="coerce")
    tickets = tickets.sort_values(
        ["_deadline_sort", "race_id", "_rule_order", "direct_ticket_score"],
        ascending=[True, True, True, False],
    )

    lines.append("【買い目だけ】")
    for race_id, g in tickets.groupby("race_id", sort=False):
        first = g.iloc[0]
        title = title_for_race(first, str(race_id))
        deadline = deadline_for_race(first)
        deadline_text = f"締切 {deadline} / " if deadline else ""
        stake = int(g["stake_yen"].sum())
        race_score = float(first.get("race_score", 0))

        lines.append(f"{title} / {deadline_text}{len(g)}点 / {stake}円 / race {race_score:.3f}")

        for r in g.itertuples():
            lines.append(
                f"  {r.combination}  {r.odds:.1f}倍  score {float(r.direct_ticket_score):.3f}  race {float(r.race_score):.3f}  {int(r.stake_yen)}円  [{short_rule_name(r.rule)}]"
            )

        lines.append("")

    lines.append("【レース詳細】")
    for race_id, g in tickets.groupby("race_id", sort=False):
        first = g.iloc[0]
        title = title_for_race(first, str(race_id))
        deadline = deadline_for_race(first)
        deadline_text = f"  締切 {deadline}" if deadline else ""
        race_score = float(first.get("race_score", 0))

        lines.append("----------------")
        lines.append(f"{title}{deadline_text}  race {race_score:.3f}")

        if str(race_id) in finish_map and finish_map[str(race_id)].strip():
            for line in finish_map[str(race_id)].splitlines():
                lines.append(simplify_finish_line(line))

        main_count = int((g["rule"] == MAIN_RULE).sum())
        high_count = int((g["rule"] == HIGH_RULE).sum())
        watch_count = int((g["rule"] == WATCH_RULE).sum())

        label_parts = []
        if main_count:
            label_parts.append(f"安定{main_count}点")
        if high_count:
            label_parts.append(f"荒れ{high_count}点")
        if watch_count:
            label_parts.append(f"大荒れ見るだけ{watch_count}点")

        if label_parts:
            lines.append("種別: " + " / ".join(label_parts))

        best = g.sort_values("direct_ticket_score", ascending=False).iloc[0]
        lines.append(
            f"最高score: {best.combination} / score {float(best.direct_ticket_score):.3f} / {float(best.odds):.1f}倍"
        )
        lines.append("")

    return "\n".join(lines).strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_date = args.target_date.strip() or today_jst()

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"target_date: {target_date}")
    print(f"dry_run: {args.dry_run}")

    raw_candidates = load_ticket_candidates(target_date)
    candidates = normalize_tickets(enrich_candidates(raw_candidates))
    finish = load_finish_predictions(target_date)

    main_tickets = select_main(candidates)
    high_tickets = select_high(candidates)
    watch_tickets = select_watch(candidates)

    tickets = pd.concat([main_tickets, high_tickets, watch_tickets], ignore_index=True, sort=False)
    if not tickets.empty:
        tickets = tickets.sort_values(["race_id", "rule", "direct_ticket_score"], ascending=[True, True, False])
        tickets = tickets.drop_duplicates(["race_id", "combination", "rule"], keep="first")

    print(f"main tickets: {len(main_tickets)}")
    print(f"high tickets: {len(high_tickets)}")
    print(f"watch tickets: {len(watch_tickets)}")

    finish_map = make_finish_summary(finish)
    summary_text = make_summary_text(target_date, tickets, finish_map)

    ticket_out = LIVE_DIR / f"live_tickets_{target_date}.csv"
    summary_out = LIVE_DIR / f"live_summary_{target_date}.txt"

    tickets.to_csv(ticket_out, index=False, encoding="utf-8-sig")
    summary_out.write_text(summary_text, encoding="utf-8")

    print("")
    print(summary_text)
    print("")
    print(f"saved: {ticket_out}")
    print(f"saved: {summary_out}")


if __name__ == "__main__":
    main()
















