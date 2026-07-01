from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


DATA_DIR = Path("data")
RAW_ROOT = DATA_DIR / "raw_kdreams_days"
DAY_ROOT = DATA_DIR / "kdreams_days"
ALL_ENTRIES_PATH = DATA_DIR / "entries_all_kdreams.csv"
ALL_PAYOUTS_PATH = DATA_DIR / "payouts_all_kdreams.csv"
ALL_FEATURES_PATH = DATA_DIR / "features_all_kdreams.csv"

STYLE_MAP = {"逃": "逃げ", "追": "追込", "両": "両", "差": "差し", "捲": "捲り"}
PREFECTURES = [
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
    "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡", "佐賀",
    "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄",
]
STYLE_COLUMNS = {"逃げ": "nige_count", "追込": "oikomi_count", "両": "ryo_count", "差し": "sashi_count", "捲り": "makuri_count"}
BAD_HTML_MARKERS = ["ページが見つかりません", "お探しのページ", "システムエラー", "ただいま混み合っています"]
RESULT_PENDING_MARKERS = ["レース結果未確定", "結果未確定"]


@dataclass(frozen=True)
class DayConfig:
    place: str
    race_date_id: str
    race_count: int
    sleep_seconds: float
    force: bool
    retries: int = 2
    retry_wait: float = 5.0
    min_html_length: int = 10_000


def race_id_for(config: DayConfig, race_no: int) -> str:
    return f"{config.race_date_id}{race_no:02d}"


def odds_url_for(config: DayConfig, race_no: int) -> str:
    race_id = race_id_for(config, race_no)
    return f"https://keirin.kdreams.jp/{config.place}/racedetail/{race_id}/?kakeshikiType=3rentan&pageType=odds"


def result_url_for(config: DayConfig) -> str:
    return f"https://keirin.kdreams.jp/{config.place}/raceresult/{config.race_date_id}/"


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urlopen(request, timeout=30) as response:
        raw = response.read()
    for encoding in ("utf-8", "cp932", "euc-jp"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def validate_html(html: str, min_html_length: int, *, result_page: bool = False) -> None:
    if len(html) < min_html_length:
        raise ValueError(f"HTMLが短すぎます: length={len(html)}")
    for marker in BAD_HTML_MARKERS:
        if marker in html:
            raise ValueError(f"エラーページらしきHTMLです: {marker}")
    if result_page:
        for marker in RESULT_PENDING_MARKERS:
            if marker in html:
                raise ValueError(f"結果未確定です: {marker}")


def fetch_cached(url: str, path: Path, config: DayConfig, *, result_page: bool = False, force_once: bool = False) -> str:
    use_cache = path.exists() and not config.force and not force_once
    if use_cache:
        html = path.read_text(encoding="utf-8")
        try:
            validate_html(html, config.min_html_length, result_page=result_page)
            print(f"  cached: {path}")
            return html
        except Exception as exc:
            print(f"  cache invalid: {exc}")

    last_error = None
    for attempt in range(config.retries + 1):
        try:
            html = fetch(url)
            validate_html(html, config.min_html_length, result_page=result_page)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            print(f"  saved: {path}")
            time.sleep(config.sleep_seconds)
            return html
        except Exception as exc:
            last_error = exc
            if attempt >= config.retries:
                break
            wait = config.retry_wait * (attempt + 1)
            print(f"  retry {attempt + 1}/{config.retries}: {exc} / wait {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"取得失敗: {url} ({last_error})")


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
        rows.append({"race_id": race_id, "car_no": car_no, "name": name, "score": to_float(row[score_col]), "style": STYLE_MAP.get(raw_style, raw_style), "age": parse_age(row[age_col]), "win_rate": to_rate(row[win_col]), "place_rate": to_rate(row[place_col])})
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("entriesが0件です")
    return output


def table_to_text(table: pd.DataFrame) -> str:
    values = []
    for _, row in table.iterrows():
        values.extend(normalize_text(value) for value in row.tolist())
    return re.sub(r"\s+", " ", " ".join(value for value in values if value)).strip()


def parse_3rentan(table: pd.DataFrame) -> tuple[str | None, int | None, int | None]:
    text = table_to_text(table)
    matches = re.findall(r"単\s+(\d+-\d+-\d+)\s+([\d,]+)円\s*\((\d+)\)", text)
    if not matches:
        return None, None, None
    combination, payout_text, popularity_text = matches[-1]
    return combination, int(payout_text.replace(",", "")), int(popularity_text)


def parse_payouts_from_result_html(html: str, config: DayConfig) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    rows = []
    race_no = 1
    for table_index in range(1, len(tables), 2):
        combination, payout, popularity = parse_3rentan(tables[table_index])
        if combination is None:
            continue
        rows.append({"race_id": race_id_for(config, race_no), "bet_type": "3連単", "combination": combination, "payout": payout, "popularity": popularity})
        race_no += 1
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("payoutsが0件です")
    return output


def make_features(entries: pd.DataFrame, payouts: pd.DataFrame) -> pd.DataFrame:
    entries = entries.copy()
    payouts = payouts.copy()
    numeric_cols = ["score", "age", "win_rate", "place_rate"]
    entries[numeric_cols] = entries[numeric_cols].apply(pd.to_numeric, errors="coerce")
    features = entries.groupby("race_id").agg(avg_score=("score", "mean"), max_score=("score", "max"), min_score=("score", "min"), std_score=("score", "std"), avg_age=("age", "mean"), racer_count=("car_no", "count"), avg_win_rate=("win_rate", "mean"), max_win_rate=("win_rate", "max"), avg_place_rate=("place_rate", "mean"), max_place_rate=("place_rate", "max"))
    features["std_score"] = features["std_score"].fillna(0)
    features["score_gap"] = features["max_score"] - features["min_score"]
    style_counts = pd.crosstab(entries["race_id"], entries["style"])
    for style, column in STYLE_COLUMNS.items():
        features[column] = style_counts.get(style, 0)
    features["front_runner_pressure"] = features["nige_count"] / features["racer_count"]
    payouts = payouts[payouts["bet_type"] == "3連単"].copy()
    payouts["payout"] = pd.to_numeric(payouts["payout"], errors="coerce")
    payouts["popularity"] = pd.to_numeric(payouts["popularity"], errors="coerce")
    payouts = payouts.rename(columns={"payout": "payout_3rentan", "popularity": "payout_popularity"})
    output = features.reset_index().merge(payouts[["race_id", "combination", "payout_3rentan", "payout_popularity"]], on="race_id", how="inner")
    output["is_over_30"] = (output["payout_3rentan"] >= 30_000).astype(int)
    output["is_over_50"] = (output["payout_3rentan"] >= 50_000).astype(int)
    output["is_over_100"] = (output["payout_3rentan"] >= 100_000).astype(int)
    output["is_popular_50plus"] = (output["payout_popularity"] >= 50).astype(int)
    output["is_popular_100plus"] = (output["payout_popularity"] >= 100).astype(int)
    output["is_popular_150plus"] = (output["payout_popularity"] >= 150).astype(int)
    if output.empty:
        raise ValueError("featuresが0件です。entriesとpayoutsのrace_idが一致していない可能性があります。")
    return output.sort_values("race_id")


def upsert_csv(path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = new_df.copy()
    for col in key_cols:
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str)

    if path.exists():
        old_df = pd.read_csv(path, dtype={col: str for col in key_cols})
        for col in key_cols:
            if col in old_df.columns:
                old_df[col] = old_df[col].astype(str)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols, key=lambda s: s.astype(str))
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined


def fetch_day(config: DayConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day_raw_dir = RAW_ROOT / config.place / config.race_date_id
    day_out_dir = DAY_ROOT / config.place / config.race_date_id
    entries_list = []
    failed_races = []
    print(f"place={config.place} race_date_id={config.race_date_id}")

    # 過去データ取得では結果が確定している日だけ処理します。
    # 未確定日を先に弾くことで、12R分のentriesを無駄に取りに行くのを避けます。
    result_url = result_url_for(config)
    result_raw_path = day_raw_dir / "result.html"
    print("result precheck")
    print(result_url)
    result_html = fetch_cached(result_url, result_raw_path, config, result_page=True)
    try:
        payouts_df = parse_payouts_from_result_html(result_html, config)
    except Exception as exc:
        print(f"  result parse failed, refetch once: {exc}")
        result_html = fetch_cached(result_url, result_raw_path, config, result_page=True, force_once=True)
        payouts_df = parse_payouts_from_result_html(result_html, config)
    print(f"  payouts: {len(payouts_df)}")

    for race_no in range(1, config.race_count + 1):
        race_id = race_id_for(config, race_no)
        url = odds_url_for(config, race_no)
        raw_path = day_raw_dir / "odds" / f"{race_id}.html"
        print(f"[{race_no}/{config.race_count}] entries {race_id}")
        print(url)
        try:
            html = fetch_cached(url, raw_path, config)
            try:
                entries = parse_entries_from_html(html, race_id)
            except Exception as exc:
                print(f"  parse failed, refetch once: {exc}")
                html = fetch_cached(url, raw_path, config, force_once=True)
                entries = parse_entries_from_html(html, race_id)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed_races.append(race_id)
            continue
        print(f"  entries: {len(entries)}")
        entries_list.append(entries)
    if not entries_list:
        raise RuntimeError("entriesを1件も取得できませんでした。")
    entries_df = pd.concat(entries_list, ignore_index=True).sort_values(["race_id", "car_no"])
    features_df = make_features(entries_df, payouts_df)
    print(f"  features: {len(features_df)}")
    if failed_races:
        print(f"  failed_races: {', '.join(failed_races)}")
    day_out_dir.mkdir(parents=True, exist_ok=True)
    entries_df.to_csv(day_out_dir / "entries.csv", index=False, encoding="utf-8-sig")
    payouts_df.to_csv(day_out_dir / "payouts.csv", index=False, encoding="utf-8-sig")
    features_df.to_csv(day_out_dir / "features.csv", index=False, encoding="utf-8-sig")
    entries_df.to_csv(DATA_DIR / "entries_from_kdreams.csv", index=False, encoding="utf-8-sig")
    payouts_df.to_csv(DATA_DIR / "payouts_from_kdreams.csv", index=False, encoding="utf-8-sig")
    features_df.to_csv(DATA_DIR / "features_from_kdreams.csv", index=False, encoding="utf-8-sig")
    all_entries = upsert_csv(ALL_ENTRIES_PATH, entries_df, ["race_id", "car_no"])
    all_payouts = upsert_csv(ALL_PAYOUTS_PATH, payouts_df, ["race_id", "bet_type"])
    all_features = upsert_csv(ALL_FEATURES_PATH, features_df, ["race_id"])
    print("saved day files:")
    print(f"  {day_out_dir / 'entries.csv'}")
    print(f"  {day_out_dir / 'payouts.csv'}")
    print(f"  {day_out_dir / 'features.csv'}")
    print("aggregate:")
    print(f"  entries_all: {len(all_entries)} rows")
    print(f"  payouts_all: {len(all_payouts)} rows")
    print(f"  features_all: {len(all_features)} rows")
    return entries_df, payouts_df, features_df


def parse_args() -> DayConfig:
    parser = argparse.ArgumentParser(description="Kドリームスから1場1日分のentries/payouts/featuresを作成します。")
    parser.add_argument("--place", default="toride")
    parser.add_argument("--race-date-id", default="23202606270100")
    parser.add_argument("--race-count", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--min-html-length", type=int, default=10_000)
    args = parser.parse_args()
    return DayConfig(args.place, args.race_date_id, args.race_count, args.sleep, args.force, args.retries, args.retry_wait, args.min_html_length)


def main() -> None:
    fetch_day(parse_args())


if __name__ == "__main__":
    main()
