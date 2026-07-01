from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_kdreams_calendar"
DEFAULT_OUTPUT = DATA_DIR / "kdreams_days_list_generated.csv"

BAD_HTML_MARKERS = ["ページが見つかりません", "お探しのページ", "システムエラー", "ただいま混み合っています"]


@dataclass(frozen=True)
class CalendarConfig:
    start_date: date
    end_date: date
    output: Path
    sleep_seconds: float
    force: bool
    only_result_links: bool
    replace_output: bool


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end は --start 以降の日付にしてください")
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def calendar_url_for(target_date: date) -> str:
    return f"https://keirin.kdreams.jp/kaisai/{target_date:%Y/%m/%d}/"


def raw_path_for(target_date: date) -> Path:
    return RAW_DIR / f"{target_date:%Y-%m-%d}.html"


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


def validate_html(html: str) -> None:
    if len(html) < 3000:
        raise ValueError(f"HTMLが短すぎます: length={len(html)}")
    for marker in BAD_HTML_MARKERS:
        if marker in html:
            raise ValueError(f"エラーページらしきHTMLです: {marker}")


def fetch_cached(target_date: date, force: bool, sleep_seconds: float) -> str:
    url = calendar_url_for(target_date)
    path = raw_path_for(target_date)
    if path.exists() and not force:
        html = path.read_text(encoding="utf-8")
        validate_html(html)
        print(f"  cached: {path}")
        return html

    html = fetch(url)
    validate_html(html)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"  saved: {path}")
    time.sleep(sleep_seconds)
    return html


def extract_days(html: str, only_result_links: bool) -> list[dict[str, object]]:
    found: dict[tuple[str, str], set[int]] = {}

    if not only_result_links:
        for place, race_id in re.findall(r"/([a-z0-9_-]+)/racedetail/(\d{16})/", html):
            race_date_id = race_id[:-2]
            race_no = int(race_id[-2:])
            found.setdefault((place, race_date_id), set()).add(race_no)

    for place, race_date_id in re.findall(r"/([a-z0-9_-]+)/raceresult/(\d{14})/", html):
        found.setdefault((place, race_date_id), set())

    rows = []
    for (place, race_date_id), race_numbers in sorted(found.items()):
        race_count = max(race_numbers) if race_numbers else 12
        rows.append({"place": place, "race_date_id": race_date_id, "race_count": race_count})
    return rows


def merge_existing(output: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    if output.exists():
        with output.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = (str(row["place"]), str(row["race_date_id"]))
                merged[key] = {"place": row["place"], "race_date_id": str(row["race_date_id"]), "race_count": int(row.get("race_count") or 12)}

    for row in rows:
        key = (str(row["place"]), str(row["race_date_id"]))
        merged[key] = {"place": row["place"], "race_date_id": str(row["race_date_id"]), "race_count": int(row.get("race_count") or 12)}

    return sorted(merged.values(), key=lambda row: (str(row["race_date_id"]), str(row["place"])))


def dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["place"]), str(row["race_date_id"]))
        merged[key] = {"place": row["place"], "race_date_id": str(row["race_date_id"]), "race_count": int(row.get("race_count") or 12)}
    return sorted(merged.values(), key=lambda row: (str(row["race_date_id"]), str(row["place"])))


def save_rows(output: Path, rows: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["place", "race_date_id", "race_count"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kドリームスの開催ページから取得対象日のCSVを作ります。")
    parser.add_argument("--date", help="1日だけ作る場合。例: 2026-06-27")
    parser.add_argument("--start", help="開始日。例: 2026-06-01")
    parser.add_argument("--end", help="終了日。例: 2026-06-27")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-result-links", action="store_true", help="結果ページへのリンクだけを対象にします")
    parser.add_argument("--replace", action="store_true", help="出力CSVを作り直します。過去の行を混ぜません")
    args = parser.parse_args()

    if args.date:
        start_date = parse_date(args.date)
        end_date = start_date
    elif args.start and args.end:
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)
    else:
        raise SystemExit("--date か、--start と --end を指定してください")

    config = CalendarConfig(start_date=start_date, end_date=end_date, output=Path(args.output), sleep_seconds=args.sleep, force=args.force, only_result_links=args.only_result_links, replace_output=args.replace)

    all_rows: list[dict[str, object]] = []
    for index, target_date in enumerate(date_range(config.start_date, config.end_date), start=1):
        print("=" * 80)
        print(f"[{index}] {target_date} {calendar_url_for(target_date)}")
        try:
            html = fetch_cached(target_date, config.force, config.sleep_seconds)
            rows = extract_days(html, config.only_result_links)
            all_rows.extend(rows)
            print(f"  found: {len(rows)} days")
            for row in rows:
                print(f"    {row['place']},{row['race_date_id']},{row['race_count']}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    merged_rows = dedupe_rows(all_rows) if config.replace_output else merge_existing(config.output, all_rows)
    save_rows(config.output, merged_rows)
    print("=" * 80)
    print(f"saved: {config.output}")
    print(f"rows: {len(merged_rows)}")


if __name__ == "__main__":
    main()
