from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

from fetch_kdreams_day import flatten_columns, normalize_key, normalize_text, pick_entry_table


DATA_DIR = Path("data")
RAW_ROOT = DATA_DIR / "raw_kdreams_days"
FEATURES_PATH = DATA_DIR / "features_all_kdreams.csv"
RACE_EXTRA_PATH = DATA_DIR / "race_extra_features.csv"
OUTPUT_PATH = DATA_DIR / "features_rich.csv"

WEATHER_PATTERN = re.compile(r"天候\s*([^/<\s]+)\s*/\s*風速\s*([0-9.]+)\s*m")


def race_info_from_path(path: Path) -> dict[str, object]:
    race_id = path.stem
    race_date_id = path.parent.parent.name
    place = path.parent.parent.parent.name
    race_no = int(race_id[-2:])
    year = int(race_id[2:6])
    month = int(race_id[6:8])
    day = int(race_id[8:10])
    held_no = int(race_id[10:12])
    day_no = int(race_id[12:14])
    return {
        "race_id": race_id,
        "race_date_id": race_date_id,
        "place": place,
        "race_no": race_no,
        "year": year,
        "month": month,
        "day": day,
        "held_no": held_no,
        "day_no": day_no,
    }


def parse_weather(html: str) -> tuple[str | None, float | None]:
    text = html.replace("\u3000", " ")
    match = WEATHER_PATTERN.search(text)
    if not match:
        return None, None
    return match.group(1), float(match.group(2))


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def find_column_soft(df: pd.DataFrame, *keywords: str) -> str | None:
    keys = [normalize_key(keyword) for keyword in keywords]
    for column in df.columns:
        key = normalize_key(column)
        if all(k in key for k in keys):
            return column
    return None


def add_col(rows: dict[str, object], prefix: str, values: pd.Series | None) -> None:
    if values is None:
        return
    nums = to_num(values)
    rows[f"avg_{prefix}"] = nums.mean()
    rows[f"max_{prefix}"] = nums.max()
    rows[f"min_{prefix}"] = nums.min()
    rows[f"sum_{prefix}"] = nums.sum()


def parse_entry_extra(html: str) -> dict[str, object]:
    tables = pd.read_html(StringIO(html))
    df = pick_entry_table(tables)
    df = flatten_columns(df)
    out: dict[str, object] = {}

    candidates = {
        "start_count": ("S",),
        "back_count": ("B",),
        "recent_nige": ("逃",),
        "recent_makuri": ("捲",),
        "recent_sashi": ("差",),
        "recent_mark": ("マ",),
        "first_count": ("1着",),
        "second_count": ("2着",),
        "third_count": ("3着",),
        "out_count": ("着外",),
        "win_rate_raw": ("勝率",),
        "quinella_rate_raw": ("2連対率",),
        "place_rate_raw": ("3連対率",),
        "gear": ("ギヤ", "倍数"),
    }

    for name, keys in candidates.items():
        col = find_column_soft(df, *keys)
        if col is not None:
            add_col(out, name, df[col])

    # 直近4ヶ月の決まり手系。荒れやすさを見るため、レース全体の偏りも作る。
    for name in ["recent_nige", "recent_makuri", "recent_sashi", "recent_mark"]:
        sum_key = f"sum_{name}"
        out.setdefault(sum_key, 0)
    move_total = sum(float(out.get(f"sum_{name}") or 0) for name in ["recent_nige", "recent_makuri", "recent_sashi", "recent_mark"])
    out["recent_move_total"] = move_total
    if move_total > 0:
        out["recent_nige_share"] = float(out.get("sum_recent_nige") or 0) / move_total
        out["recent_makuri_share"] = float(out.get("sum_recent_makuri") or 0) / move_total
        out["recent_sashi_share"] = float(out.get("sum_recent_sashi") or 0) / move_total
        out["recent_mark_share"] = float(out.get("sum_recent_mark") or 0) / move_total
    else:
        out["recent_nige_share"] = 0
        out["recent_makuri_share"] = 0
        out["recent_sashi_share"] = 0
        out["recent_mark_share"] = 0

    return out


def main() -> None:
    features = pd.read_csv(FEATURES_PATH, dtype={"race_id": str})
    html_files = list(RAW_ROOT.glob("*/*/odds/*.html"))
    print(f"odds html files: {len(html_files):,}")

    rows = []
    for index, html_path in enumerate(html_files, start=1):
        if index % 5000 == 0:
            print(f"  parsed {index:,}/{len(html_files):,}")
        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            row = race_info_from_path(html_path)
            weather, wind_speed = parse_weather(html)
            row["weather"] = weather
            row["wind_speed"] = wind_speed
            row.update(parse_entry_extra(html))
            rows.append(row)
        except Exception as exc:
            info = race_info_from_path(html_path)
            rows.append({**info, "parse_error": repr(exc)})

    extra = pd.DataFrame(rows).drop_duplicates(subset=["race_id"], keep="last")
    extra.to_csv(RACE_EXTRA_PATH, index=False, encoding="utf-8-sig")

    rich = features.merge(extra, on="race_id", how="left")
    rich["weather"] = rich["weather"].fillna("unknown")
    rich["wind_speed"] = pd.to_numeric(rich["wind_speed"], errors="coerce")
    rich["is_rain"] = rich["weather"].astype(str).str.contains("雨", na=False).astype(int)
    rich["is_snow"] = rich["weather"].astype(str).str.contains("雪", na=False).astype(int)
    rich["is_cloudy"] = rich["weather"].astype(str).str.contains("曇", na=False).astype(int)
    rich["is_sunny"] = rich["weather"].astype(str).str.contains("晴", na=False).astype(int)
    rich["is_strong_wind"] = (rich["wind_speed"] >= 3.0).astype(int)
    rich["is_very_strong_wind"] = (rich["wind_speed"] >= 5.0).astype(int)

    numeric_fill_cols = [c for c in rich.columns if c.startswith(("avg_", "max_", "min_", "sum_")) or c.startswith("recent_")]
    rich[numeric_fill_cols] = rich[numeric_fill_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    rich.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"extra rows: {len(extra):,}")
    print(f"features rows: {len(rich):,}")
    print(f"weather missing/unknown: {(rich['weather'] == 'unknown').sum():,}")
    print(f"saved: {RACE_EXTRA_PATH}")
    print(f"saved: {OUTPUT_PATH}")
    print(rich[["race_id", "place", "race_no", "weather", "wind_speed", "sum_recent_nige", "sum_recent_makuri", "sum_recent_sashi", "sum_recent_mark"]].head())


if __name__ == "__main__":
    main()
