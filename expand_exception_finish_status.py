from pathlib import Path
import re
from html import unescape
import pandas as pd

DATA = Path("data")
RAW_DIR = DATA / "raw_kdreams_days"
IN_CLEAN = DATA / "race_results_full_clean.csv"
OUT = DATA / "race_results_full_clean_v2.csv"
OUT_EX = DATA / "race_results_exception_details.csv"

TAG_RE = re.compile(r"<[^>]+>")
TABLE_RE = re.compile(r'<table[^>]*class="[^"]*result_table[^"]*"[^>]*>(.*?)</table>', re.I | re.S)
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.I | re.S)

def clean(s):
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = TAG_RE.sub("", s)
    s = unescape(s).replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm(s):
    return re.sub(r"\s+", "", s)

def classify(raw):
    text = str(raw)
    if "欠" in text:
        return "absent"
    if "失" in text:
        return "disqualified"
    if "落" in text and "棄" in text:
        return "crash_dnf"
    if "落" in text:
        return "crash"
    if "棄" in text:
        return "dnf"
    if "事故" in text or "故" in text or "再" in text:
        return "accident_finish"
    return "unknown_exception"

def parse_exception_rows(path, race_id):
    html = path.read_text(encoding="utf-8", errors="ignore")
    out = []

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

        for row in rows[1:]:
            raw_pos = get(row, "着順")
            if str(raw_pos).isdigit():
                continue

            car = get(row, "車番")
            m = re.search(r"\d+", str(car))
            if not m:
                continue

            status = classify(raw_pos)
            out.append({
                "race_id": race_id,
                "car_no": int(m.group(0)),
                "raw_finish_position": raw_pos,
                "finish_status_detail": status,
                "exception_name_from_html": get(row, "選手名").replace(" ", ""),
                "exception_margin": get(row, "着差"),
                "exception_agari": get(row, "上り"),
                "exception_decision": get(row, "決まり手"),
                "exception_sb": get(row, "S／B"),
                "exception_comment": get(row, "勝敗因"),
            })

        return out

    return out

def main():
    print("loading clean csv...")
    df = pd.read_csv(IN_CLEAN, dtype={"race_id": str})
    df["race_id"] = df["race_id"].astype(str).str.zfill(16)
    df["car_no"] = pd.to_numeric(df["car_no"], errors="coerce").astype("Int64")

    target_races = set(df.loc[df["finish_status"] == "no_numeric_finish", "race_id"].unique())
    print(f"exception target races: {len(target_races):,}")

    print("indexing html files...")
    file_map = {}
    for p in RAW_DIR.glob("*/*/odds/*.html"):
        race_id = p.stem.zfill(16)
        if race_id in target_races:
            file_map[race_id] = p

    print(f"html found: {len(file_map):,}")

    ex_rows = []
    for i, (race_id, path) in enumerate(file_map.items(), 1):
        ex_rows.extend(parse_exception_rows(path, race_id))
        if i % 1000 == 0:
            print(f"processed exception html: {i:,}/{len(file_map):,} / exception rows={len(ex_rows):,}")

    ex = pd.DataFrame(ex_rows)
    if len(ex) == 0:
        print("no exception details found")
        df.to_csv(OUT, index=False, encoding="utf-8-sig")
        return

    ex["race_id"] = ex["race_id"].astype(str).str.zfill(16)
    ex["car_no"] = pd.to_numeric(ex["car_no"], errors="coerce").astype("Int64")
    ex = ex.drop_duplicates(["race_id", "car_no"])

    df = df.merge(ex, on=["race_id", "car_no"], how="left")

    df["finish_status_detail"] = df["finish_status_detail"].fillna(df["finish_status"])
    mask = df["finish_status"].eq("no_numeric_finish") & df["finish_status_detail"].eq("no_numeric_finish")
    df.loc[mask, "finish_status_detail"] = "unknown_exception"

    df["is_absent"] = df["finish_status_detail"].eq("absent").astype(int)
    df["is_dnf"] = df["finish_status_detail"].isin(["dnf", "crash_dnf"]).astype(int)
    df["is_crash"] = df["finish_status_detail"].isin(["crash", "crash_dnf"]).astype(int)
    df["is_disqualified"] = df["finish_status_detail"].eq("disqualified").astype(int)
    df["is_accident_finish"] = df["finish_status_detail"].eq("accident_finish").astype(int)
    df["is_unknown_exception"] = df["finish_status_detail"].eq("unknown_exception").astype(int)

    ex.to_csv(OUT_EX, index=False, encoding="utf-8-sig")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT}")
    print(f"saved: {OUT_EX}")
    print("")
    print("finish_status_detail:")
    print(df["finish_status_detail"].value_counts(dropna=False).to_string())
    print("")
    print("exception details:")
    print(ex["finish_status_detail"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
