from __future__ import annotations

import argparse
import re
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


def parse_file(path):
    race_id = path.stem
    html = path.read_text(encoding="utf-8", errors="ignore")
    block = extract_3rentan_block(html)
    if not block:
        return []

    try:
        tables = pd.read_html(StringIO(block))
    except ValueError:
        return []

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

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(OUT))
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    files = sorted(RAW_ROOT.glob("*/**/odds/*.html"))
    if args.limit:
        files = files[:args.limit]

    all_rows = []
    failed = []

    print(f"html files: {len(files):,}")

    for i, path in enumerate(files, 1):
        rows = parse_file(path)
        if rows:
            all_rows.extend(rows)
        else:
            failed.append(str(path))

        if i == 1 or i % args.progress_every == 0:
            print(f"processed: {i:,}/{len(files):,} / odds rows={len(all_rows):,} / failed={len(failed):,}")

    df = pd.DataFrame(all_rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    failed_path = out.with_name(out.stem + "_failed.txt")
    failed_path.write_text("\n".join(failed), encoding="utf-8")

    print(f"saved: {out}")
    print(f"rows: {len(df):,}")
    print(f"races: {df['race_id'].nunique() if not df.empty else 0:,}")
    print(f"failed: {len(failed):,}")
    print(df.head(20).to_string(index=False) if not df.empty else "empty")


if __name__ == "__main__":
    main()
