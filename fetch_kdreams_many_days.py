from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

from fetch_kdreams_day import DAY_ROOT, DayConfig, fetch_day


DATA_DIR = Path("data")
DEFAULT_DAYS_FILE = DATA_DIR / "kdreams_days_list.csv"
LOG_PATH = DATA_DIR / "fetch_logs_kdreams.csv"


def ensure_default_days_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"place": "toride", "race_date_id": "23202606270100", "race_count": 12},
        {"place": "toride", "race_date_id": "23202606270200", "race_count": 12},
        {"place": "toride", "race_date_id": "23202606270300", "race_count": 12},
        {"place": "toride", "race_date_id": "23202606270400", "race_count": 12},
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["place", "race_date_id", "race_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"日付リストの雛形を作成しました: {path}")


def load_days(path: Path) -> list[dict[str, str]]:
    ensure_default_days_file(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_log(row: dict[str, object]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    fieldnames = ["timestamp", "place", "race_date_id", "race_count", "status", "entries_rows", "payouts_rows", "features_rows", "error"]
    with LOG_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def latest_status_map() -> dict[tuple[str, str], str]:
    if not LOG_PATH.exists():
        return {}
    logs = pd.read_csv(LOG_PATH)
    if logs.empty:
        return {}
    latest = logs.drop_duplicates(subset=["place", "race_date_id"], keep="last")
    return {(str(row.place), str(row.race_date_id)): str(row.status) for row in latest.itertuples(index=False)}


def day_features_exists(place: str, race_date_id: str) -> bool:
    path = DAY_ROOT / place / race_date_id / "features.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        return not df.empty
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Kドリームスの複数日分を順番に取得します。")
    parser.add_argument("--days-file", default=str(DEFAULT_DAYS_FILE))
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--min-html-length", type=int, default=10_000)
    parser.add_argument("--skip-success", action="store_true", help="成功済みの日をスキップします")
    parser.add_argument("--only-failed", action="store_true", help="最新ログがerrorの日だけ再実行します")
    args = parser.parse_args()
    rows = load_days(Path(args.days_file))
    status_map = latest_status_map()
    if args.only_failed:
        rows = [row for row in rows if status_map.get((row["place"].strip(), row["race_date_id"].strip())) == "error"]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"days_file={args.days_file}")
    print(f"targets={len(rows)}")
    success_count = 0
    failure_count = 0
    skipped_count = 0
    for index, row in enumerate(rows, start=1):
        place = row["place"].strip()
        race_date_id = row["race_date_id"].strip()
        race_count = int(row.get("race_count") or 12)
        if args.skip_success and not args.force and day_features_exists(place, race_date_id):
            print("=" * 80)
            print(f"[{index}/{len(rows)}] skip success {place} {race_date_id}")
            skipped_count += 1
            continue
        config = DayConfig(place=place, race_date_id=race_date_id, race_count=race_count, sleep_seconds=args.sleep, force=args.force, retries=args.retries, retry_wait=args.retry_wait, min_html_length=args.min_html_length)
        print("=" * 80)
        print(f"[{index}/{len(rows)}] {place} {race_date_id}")
        log_row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "place": place, "race_date_id": race_date_id, "race_count": race_count, "status": "ok", "entries_rows": 0, "payouts_rows": 0, "features_rows": 0, "error": ""}
        try:
            entries, payouts, features = fetch_day(config)
            log_row["entries_rows"] = len(entries)
            log_row["payouts_rows"] = len(payouts)
            log_row["features_rows"] = len(features)
            success_count += 1
        except Exception as exc:
            log_row["status"] = "error"
            log_row["error"] = repr(exc)
            failure_count += 1
            print(f"ERROR: {exc}")
        finally:
            append_log(log_row)
    print("=" * 80)
    print(f"done: success={success_count}, failure={failure_count}, skipped={skipped_count}")
    print(f"log: {LOG_PATH}")


if __name__ == "__main__":
    main()
