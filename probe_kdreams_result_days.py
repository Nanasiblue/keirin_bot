from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from fetch_kdreams_day import DayConfig, RAW_ROOT, parse_payouts_from_result_html, result_url_for

DATA_DIR = Path("data")
DEFAULT_SOURCE = DATA_DIR / "kdreams_days_list_2023_20260626_unique.csv"
DEFAULT_OUTPUT = DATA_DIR / "kdreams_days_list_final_test.csv"
BAD_MARKERS = ["ページが見つかりません", "お探しのページ", "システムエラー", "ただいま混み合っています", "レース結果未確定", "結果未確定"]


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


def date_range(start: str, end: str) -> list[datetime]:
    current = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")
    days = []
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days


def load_place_codes(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            place = row["place"].strip()
            race_date_id = str(row["race_date_id"]).strip()
            if len(race_date_id) >= 2:
                rows.append((place, race_date_id[:2]))
    return sorted(set(rows), key=lambda x: (x[1], x[0]))


def has_result_tables(html: str) -> bool:
    if len(html) < 5000:
        return False
    if any(marker in html for marker in BAD_MARKERS):
        return False
    if "3連単" not in html:
        return False
    try:
        tables = pd.read_html(StringIO(html))
        return len(tables) > 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="結果ページ候補を確認し、取得できる開催日CSVを作ります。")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-day-no", type=int, default=4)
    args = parser.parse_args()

    place_codes = load_place_codes(Path(args.source))
    found = []
    checked = 0
    target_days = date_range(args.start, args.end)

    for target_day in target_days:
        print("=" * 80)
        print(f"target date: {target_day:%Y-%m-%d}")
        # race_date_id の日付部分は開催初日のことが多いので、対象日から数日前まで戻して候補を作る。
        start_day_candidates = [target_day - timedelta(days=offset) for offset in range(args.max_day_no)]
        for place, code in place_codes:
            for start_day in start_day_candidates:
                actual_day_no = (target_day - start_day).days + 1
                if actual_day_no < 1 or actual_day_no > args.max_day_no:
                    continue
                race_date_id = f"{code}{start_day:%Y%m%d}{actual_day_no:02d}00"
                config = DayConfig(place=place, race_date_id=race_date_id, race_count=12, sleep_seconds=0, force=False)
                url = result_url_for(config)
                checked += 1
                try:
                    html = fetch(url)
                    if not has_result_tables(html):
                        continue
                    payouts = parse_payouts_from_result_html(html, config)
                    race_count = len(payouts)
                    if race_count <= 0:
                        continue
                    result_path = RAW_ROOT / place / race_date_id / "result.html"
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(html, encoding="utf-8")
                    found.append({"place": place, "race_date_id": race_date_id, "race_count": race_count})
                    print(f"  found: {place},{race_date_id},{race_count}")
                    break
                except Exception:
                    continue
                finally:
                    if args.sleep > 0:
                        time.sleep(args.sleep)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    unique = {(row["place"], row["race_date_id"]): row for row in found}
    rows = sorted(unique.values(), key=lambda r: (r["race_date_id"], r["place"]))
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["place", "race_date_id", "race_count"])
        writer.writeheader()
        writer.writerows(rows)
    print("=" * 80)
    print(f"checked: {checked}")
    print(f"saved: {output}")
    print(f"rows: {len(rows)}")

if __name__ == "__main__":
    main()
