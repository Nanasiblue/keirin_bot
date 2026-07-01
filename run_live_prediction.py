from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


LIVE_DIR = Path("data/live_predictions")
LOG_DIR = Path("data/live_logs")
LIVE_INPUT_DIR = Path("data/live_input")


def today_jst() -> str:
    return (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d")


def run_command(cmd: list[str], log_path: Path, required: bool = True) -> int:
    with log_path.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        proc.wait()
        f.write(f"\nexit_code={proc.returncode}\n\n")

    if required and proc.returncode != 0:
        raise SystemExit(proc.returncode)

    return int(proc.returncode)


def write_status(target_date: str, status: str, message: str) -> Path:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    out = LIVE_DIR / f"live_status_{target_date}.csv"
    pd.DataFrame(
        [
            {
                "target_date": target_date,
                "status": status,
                "message": message,
                "created_at_jst": today_jst(),
            }
        ]
    ).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", default="")
    parser.add_argument("--dry-run", default="true")
    args = parser.parse_args()

    target_date = args.target_date.strip() or today_jst()
    jst_today = today_jst()
    dry_run = str(args.dry_run).lower() not in {"false", "0", "no"}

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"live_prediction_{target_date}.log"

    print(f"target_date: {target_date}")
    print(f"jst_today: {jst_today}")
    print(f"dry_run: {dry_run}")
    print(f"log: {log_path}")

    try:
        # 1. 今日の未締切レースをKDreamsから取得
        # probe_live_kdreams.py は「現在JSTの日付」で open_races_YYYY-MM-DD.csv を作る。
        if target_date == jst_today:
            if Path("probe_live_kdreams.py").exists():
                run_command([sys.executable, "probe_live_kdreams.py"], log_path, required=True)
            else:
                raise SystemExit("probe_live_kdreams.py がありません")
        else:
            print("target_date is not today. Skip live scraping and use existing input if available.")

        open_races = LIVE_INPUT_DIR / f"open_races_{target_date}.csv"

        # 2. HTMLからentries/oddsを作り、モデルでAI入力候補を作る
        if open_races.exists() and Path("build_live_input_from_kdreams.py").exists():
            run_command(
                [sys.executable, "build_live_input_from_kdreams.py", "--target-date", target_date],
                log_path,
                required=True,
            )
        else:
            print(f"skip build_live_input_from_kdreams.py: {open_races} not found")

        # 3. 予想CSVとDiscord用summaryを作る
        if Path("predict_live_day.py").exists():
            cmd = [sys.executable, "predict_live_day.py", "--target-date", target_date]
            if dry_run:
                cmd.append("--dry-run")
            run_command(cmd, log_path, required=True)
        else:
            raise SystemExit("predict_live_day.py がありません")

        # 4. Discord通知。Webhook未設定ならスキップ。
        if Path("notify_discord.py").exists() and os.environ.get("DISCORD_WEBHOOK_URL", "").strip():
            run_command(
                [sys.executable, "notify_discord.py", "--target-date", target_date],
                log_path,
                required=True,
            )
        else:
            print("skip Discord notify: notify_discord.py or DISCORD_WEBHOOK_URL is missing")

        write_status(target_date, "ok", "live prediction completed")
        print("live prediction completed")

    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
        write_status(target_date, "failed", f"live prediction failed: exit_code={code}")
        raise

    except Exception as e:
        write_status(target_date, "failed", f"live prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
