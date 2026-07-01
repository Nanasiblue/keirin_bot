from pathlib import Path
import re
import time
from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_kdreams_entries"
OUTPUT_CSV = DATA_DIR / "entries_from_kdreams.csv"

PLACE_SLUG = "toride"
RACE_DATE_ID = "23202606270100"
RACE_COUNT = 12
SLEEP_SECONDS = 1.0

STYLE_MAP = {"逃": "逃げ", "追": "追込", "両": "両", "差": "差し", "捲": "捲り"}
PREFECTURES = [
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
    "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡", "佐賀",
    "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄",
]


def race_id_for(race_no: int) -> str:
    return f"{RACE_DATE_ID}{race_no:02d}"


def odds_url_for(race_no: int) -> str:
    race_id = race_id_for(race_no)
    return f"https://keirin.kdreams.jp/{PLACE_SLUG}/racedetail/{race_id}/?kakeshikiType=3rentan&pageType=odds"


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urlopen(request, timeout=20) as response:
        raw = response.read()
    for encoding in ("utf-8", "cp932", "euc-jp"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\u3000", " ").strip()


def compact_text(value: object) -> str:
    return normalize_text(value).replace(" ", "")


def normalize_key(value: object) -> str:
    return compact_text(value).replace("_", "")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    copied = df.copy()
    if isinstance(copied.columns, pd.MultiIndex):
        columns = []
        for column in copied.columns:
            parts = []
            for part in column:
                text = normalize_text(part)
                if text and not text.startswith("Unnamed") and text not in parts:
                    parts.append(text)
            columns.append("_".join(parts))
        copied.columns = columns
    else:
        copied.columns = [normalize_text(col) for col in copied.columns]
    return copied


def find_column(df: pd.DataFrame, *keywords: str) -> str:
    keys = [normalize_key(keyword) for keyword in keywords]
    for column in df.columns:
        key = normalize_key(column)
        if all(keyword in key for keyword in keys):
            return column
    raise KeyError(f"列が見つかりません: {keywords}\ncolumns={list(df.columns)}")


def pick_entry_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        df = flatten_columns(table)
        joined = " ".join(normalize_key(col) for col in df.columns)
        if "車番" in joined and "選手名" in joined and "競走得点" in joined and "勝率" in joined:
            return df
    raise ValueError("選手成績テーブルが見つかりませんでした。")


def parse_name(value: object) -> str:
    text = compact_text(value)
    for prefecture in sorted(PREFECTURES, key=len, reverse=True):
        marker = f"{prefecture}/"
        index = text.find(marker)
        if index > 0:
            return text[:index]
    return re.sub(r"/.+$", "", text)


def parse_age(value: object) -> int | None:
    match = re.search(r"/(\d{1,3})/", normalize_text(value))
    return int(match.group(1)) if match else None


def to_rate(value: object) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return round(float(text) / 100.0, 4)
    except ValueError:
        return None


def to_float(value: object) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: object) -> int | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_entries_from_html(html: str, race_id: str) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_entries = []

    for race_no in range(1, RACE_COUNT + 1):
        race_id = race_id_for(race_no)
        url = odds_url_for(race_no)
        raw_path = RAW_DIR / f"{race_id}.html"
        print(f"[{race_no}/{RACE_COUNT}] {race_id}")
        print(url)

        if raw_path.exists():
            html = raw_path.read_text(encoding="utf-8")
            print(f"  cached: {raw_path}")
        else:
            html = fetch(url)
            raw_path.write_text(html, encoding="utf-8")
            print(f"  saved: {raw_path}")
            time.sleep(SLEEP_SECONDS)

        try:
            entries = parse_entries_from_html(html, race_id)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        print(f"  entries: {len(entries)}")
        all_entries.append(entries)

    if not all_entries:
        raise RuntimeError("entriesを1件も取得できませんでした。")

    output = pd.concat(all_entries, ignore_index=True).sort_values(["race_id", "car_no"])
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"entries CSVを作成しました: {OUTPUT_CSV}")
    print(f"races: {output['race_id'].nunique()}, rows: {len(output)}")
    print(output.head(20))


if __name__ == "__main__":
    main()
