from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import pandas as pd


LIVE_DIR = Path("data/live_predictions")
LOG_DIR = Path("data/live_logs")


def today_jst() -> str:
    return (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d")


def run_command(cmd: list[str], log_path: Path) -> int:
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
        f.write(f"\nexit_code={proc.returncode}\n")
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
    dry_run = str(args.dry_run).lower() not in {"false", "0", "no"}

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"live_prediction_{target_date}.log"

    print(f"target_date: {target_date}")
    print(f"dry_run: {dry_run}")
    print(f"log: {log_path}")

    live_script = Path("predict_live_day.py")
    if live_script.exists():
        cmd = [sys.executable, str(live_script), "--target-date", target_date]
        if dry_run:
            cmd.append("--dry-run")
        code = run_command(cmd, log_path)
        if code != 0:
            write_status(target_date, "failed", f"predict_live_day.py failed: exit_code={code}")
            raise SystemExit(code)
        write_status(target_date, "ok", "predict_live_day.py completed")
        return

    msg = (
        "GitHub Actions plumbing is ready. "
        "Next: create predict_live_day.py to fetch live odds and output live tickets."
    )
    out = write_status(target_date, "setup_only", msg)
    print(msg)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
