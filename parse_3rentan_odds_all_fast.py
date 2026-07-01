from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import pandas as pd


RAW_ROOT = Path("data/raw_kdreams_days")
OUT = Path("data/odds_3rentan_all.csv")


def as_int(value):
    if pd.isna(value):
        return None
    m = re.search(r"\d+", str(value).strip())
    return int(m.group(0)) if m else None


def as_float(value):
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_first_car(table):
    col = table.columns[0]
    text = " ".join(str(x) for x in col if not str(x).startswith("Unnamed")) if isinstance(col, tuple) else str(col)
    m = re.search(r"^\s*(\d+)", text)
    return int(m.group(1)) if m else None


def parse_second_car(column):
    values = list(column)[::-1] if isinstance(column, tuple) else [column]
    for v in values:
        text = str(v).strip()
        if text.startswith("Unnamed"):
            continue
        if re.fullmatch(r"\d+", text):
            return int(text)
    return None


def extract_3rentan_block(html):
    start = html.find("JS_ODDSCONTENTS_3rentan")
    if start < 0:
        return ""
    m = re.search(r'id="JS_ODDSCONTENTS_[^"]+"', html[start + 1:])
    if not m:
        return html[start:]
    return html[start:start + 1 + m.start()]


def parse_file(path_text):
    path = Path(path_text)
    race_id = path.stem

    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        block = extract_3rentan_block(html)
        if not block:
            return [], str(path)

        tables = pd.read_html(StringIO(block))
    except Exception:
        return [], str(path)

    rows = []

    for table in tables:
        first_car = parse_first_car(table)
        if first_car is None:
            continue

        for col_i, col in enumerate(table.columns):
            second_car = parse_second_car(col)
            if second_car is None:
                continue

            for _, row in table.iterrows():
                third_car = as_int(row.iloc[0])
                odds = as_float(row.iloc[col_i])

                if third_car is None or odds is None:
                    continue
                if len({first_car, second_car, third_car}) != 3:
                    continue

                rows.append({
                    "race_id": race_id,
                    "combination": f"{first_car}-{second_car}-{third_car}",
                    "odds": odds,
                    "estimated_payout": int(round(odds * 100)),
                })

    if not rows:
        return [], str(path)

    return rows, None


def write_rows(output, rows, write_header):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(
        output,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        encoding="utf-8-sig" if write_header else "utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--flush-files", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(OUT))
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()

    output = Path(args.output)
    failed_path = output.with_name(output.stem + "_failed.txt")

    if output.exists() and not args.replace:
        raise SystemExit(f"{output} already exists. 上書きするなら --replace を付けてください。")

    if output.exists() and args.replace:
        output.unlink()
    if failed_path.exists() and args.replace:
        failed_path.unlink()

    files = sorted(str(p) for p in RAW_ROOT.glob("*/**/odds/*.html"))
    if args.limit:
        files = files[:args.limit]

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"html files: {len(files):,}")
    print(f"workers: {args.workers}")
    print(f"output: {output}")

    total_rows = 0
    done = 0
    failed = []
    buffer = []
    buffered_files = 0
    write_header = True

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(parse_file, path) for path in files]

        for future in as_completed(futures):
            rows, fail = future.result()
            done += 1
            buffered_files += 1

            if rows:
                buffer.extend(rows)
                total_rows += len(rows)
            if fail:
                failed.append(fail)

            if buffered_files >= args.flush_files:
                write_rows(output, buffer, write_header)
                write_header = False
                buffer = []
                buffered_files = 0
                print(f"processed: {done:,}/{len(files):,} / odds rows={total_rows:,} / failed={len(failed):,}")

    write_rows(output, buffer, write_header)

    failed_path.write_text("\n".join(failed), encoding="utf-8")

    print("done")
    print(f"saved: {output}")
    print(f"rows: {total_rows:,}")
    print(f"failed: {len(failed):,}")
    print(f"failed list: {failed_path}")


if __name__ == "__main__":
    main()
