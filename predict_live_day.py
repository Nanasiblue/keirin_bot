from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd


LIVE_DIR = Path("data/live_predictions")
LOG_DIR = Path("data/live_logs")

HIST_TICKETS = Path("data/direct_ticket_predictions_full_valid.csv")
HIST_FINISH = Path("data/finish_lightgbm_predictions_valid.csv")
RACE_SCORE_FILE = Path("data/predictions_valid_2026_is_over_50.csv")
CONTEXT_FILES = [
    Path("data/features_rich.csv"),
    Path("data/features_all_kdreams.csv"),
    Path("data/features_from_kdreams.csv"),
]

MAIN_RULE = "direct_v1_middle_odds"
HIGH_RULE = "high_odds_v1_top5"


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

    if RACE_SCORE_FILE.exists():
        score = read_csv_if_exists(RACE_SCORE_FILE)
        score_col = detect_score_col(score)
        if score_col:
            score = score[["race_id", score_col]].rename(columns={score_col: "race_score"})
            parts.append(score)
            print(f"race_score loaded: {RACE_SCORE_FILE} / column={score_col}")
        else:
            print(f"WARNING: race_score column not found in {RACE_SCORE_FILE}")

    for path in CONTEXT_FILES:
        if not path.exists():
            continue
        ctx = read_csv_if_exists(path)
        keep = [c for c in ["race_id", "place", "race_no", "grade", "weather", "wind_speed"] if c in ctx.columns]
        if "race_id" in keep and len(keep) > 1:
            parts.append(ctx[keep].drop_duplicates("race_id"))
            print(f"context loaded: {path} / columns={keep}")

    if not parts:
        return pd.DataFrame(columns=["race_id"])

    out = parts[0].drop_duplicates("race_id").copy()
    for p in parts[1:]:
        out = out.merge(p.drop_duplicates("race_id"), on="race_id", how="outer", suffixes=("", "_ctx"))
        for col in ["place", "race_no", "grade", "weather", "wind_speed", "race_score"]:
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

    for col in ["place", "race_no", "grade", "weather", "wind_speed", "race_score"]:
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
        (df["odds"] >= 30)
        & (df["odds"] <= 500)
        & (df["direct_ticket_score"] >= 0.90)
    ].copy()

    if base.empty:
        return base

    selected = (
        base.sort_values(["race_id", "direct_ticket_score"], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(2)
        .copy()
    )
    selected["rule"] = MAIN_RULE
    selected["stake_yen"] = 100
    selected["reason"] = "本線: 中オッズ + 高スコア"
    return selected


def select_high(df: pd.DataFrame) -> pd.DataFrame:
    base = df[
        (df["odds"] >= 100)
        & (df["odds"] <= 500)
        & (df["race_score"] >= 0.45)
        & (df["direct_ticket_score"] >= 0.40)
        & (df["direct_expected_return"] >= 3000)
    ].copy()

    if base.empty:
        return base

    selected = (
        base.sort_values(["race_id", "direct_expected_return"], ascending=[True, False])
        .groupby("race_id", group_keys=False)
        .head(5)
        .copy()
    )
    selected["rule"] = HIGH_RULE
    selected["stake_yen"] = 100
    selected["reason"] = "荒れサブ: 高オッズ + 荒れスコア"
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


def make_summary_text(target_date: str, tickets: pd.DataFrame, finish_map: dict[str, str]) -> str:
    lines = []
    lines.append(f"競輪AI 予想 {target_date}")
    lines.append("")
    lines.append(f"買い目数: {len(tickets)}")
    lines.append(f"対象レース: {tickets['race_id'].nunique() if not tickets.empty else 0}")
    lines.append(f"想定購入額: {int(tickets['stake_yen'].sum()) if not tickets.empty else 0}円")
    lines.append("")

    if tickets.empty:
        lines.append("本日の買い目候補はありません。")
        return "\n".join(lines)

    for race_id, g in tickets.sort_values(["race_id", "rule", "direct_ticket_score"], ascending=[True, True, False]).groupby("race_id"):
        first = g.iloc[0]
        lines.append(f"【{title_for_race(first, str(race_id))}】")

        if str(race_id) in finish_map:
            lines.append(finish_map[str(race_id)])

        for r in g.itertuples():
            lines.append(
                f"- {r.rule}: {r.combination} / {r.odds:.1f}倍 / "
                f"score {r.direct_ticket_score:.3f} / race {r.race_score:.3f} / {int(r.stake_yen)}円"
            )
        lines.append("")

    return "\n".join(lines)


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

    tickets = pd.concat([main_tickets, high_tickets], ignore_index=True, sort=False)
    if not tickets.empty:
        tickets = tickets.sort_values(["race_id", "rule", "direct_ticket_score"], ascending=[True, True, False])
        tickets = tickets.drop_duplicates(["race_id", "combination", "rule"], keep="first")

    print(f"main tickets: {len(main_tickets)}")
    print(f"high tickets: {len(high_tickets)}")

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
