from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import requests


def today_jst() -> str:
    return (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", default="")
    args = parser.parse_args()

    target_date = args.target_date.strip() or today_jst()
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

    if not webhook:
        print("DISCORD_WEBHOOK_URL is empty. skip discord notify.")
        return

    summary_path = Path(f"data/live_predictions/live_summary_{target_date}.txt")
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")

    text = summary_path.read_text(encoding="utf-8").strip()
    if not text:
        print("summary is empty. skip discord notify.")
        return

    chunks = []
    while text:
        chunks.append(text[:1800])
        text = text[1800:]

    for i, chunk in enumerate(chunks, start=1):
        content = chunk
        if len(chunks) > 1:
            content = f"{chunk}\n\n({i}/{len(chunks)})"

        res = requests.post(webhook, json={"content": content}, timeout=30)
        print(f"discord status: {res.status_code}")
        res.raise_for_status()


if __name__ == "__main__":
    main()
