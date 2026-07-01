from __future__ import annotations

import csv
import re
from html import unescape
from pathlib import Path

RAW_DIR = Path("data/raw_kdreams_days")
OUT = Path("data/race_results_full.csv")

TAG_RE = re.compile(r"<[^>]+>")
TABLE_RE = re.compile(r'<table[^>]*class="[^"]*\bresult_table\b[^"]*"[^>]*>(.*?)</table>', re.I | re.S)
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.I | re.S)

def clean(s):
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = TAG_RE.sub("", s)
    s = unescape(s).replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm(s):
    return re.sub(r"\s+", "", s)

def parse_file(path):
    race_id = path.stem
    html = path.read_text(encoding="utf-8", errors="ignore")

    for table in TABLE_RE.findall(html):
        rows = []
        for tr in TR_RE.findall(table):
            cells = [clean(c) for c in CELL_RE.findall(tr)]
            if cells:
                rows.append(cells)

        if not rows:
            continue

        header = [norm(c) for c in rows[0]]
        if not {"着順", "車番", "選手名"}.issubset(set(header)):
            continue

        idx = {name: i for i, name in enumerate(header)}

        def get(row, col):
            i = idx.get(col)
            if i is None or i >= len(row):
                return ""
            return row[i]

        out = []
        for row in rows[1:]:
            pos = get(row, "着順")
            car = get(row, "車番")
            if not pos.isdigit():
                continue
            m = re.search(r"\d+", car)
            if not m:
                continue

            out.append({
                "race_id": race_id,
                "finish_position": int(pos),
                "car_no": int(m.group(0)),
                "name": get(row, "選手名").replace(" ", ""),
                "margin": get(row, "着差"),
                "agari": get(row, "上り"),
                "decision": get(row, "決まり手"),
                "sb": get(row, "S／B"),
                "comment": get(row, "勝敗因"),
            })

        return out

    return []

def main():
    files = sorted(RAW_DIR.glob("*/*/odds/*.html"))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "race_id", "finish_position", "car_no", "name",
        "margin", "agari", "decision", "sb", "comment",
    ]

    total = 0
    parsed = 0

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(files, 1):
            rows = parse_file(path)
            if rows:
                writer.writerows(rows)
                total += len(rows)
                parsed += 1

            if i % 5000 == 0:
                print(f"processed: {i:,}/{len(files):,} files / rows={total:,}")

    print(f"html files: {len(files):,}")
    print(f"parsed files: {parsed:,}")
    print(f"rows: {total:,}")
    print(f"saved: {OUT}")

if __name__ == "__main__":
    main()
