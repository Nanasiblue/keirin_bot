from pathlib import Path
import re
from io import StringIO

import pandas as pd


INPUT_HTML = Path("data") / "raw_probe" / "result.html"
OUTPUT_CSV = Path("data") / "payouts_from_kdreams.csv"
DEFAULT_RACE_DATE_ID = "23202606270100"


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\u3000", " ").strip()


def table_to_text(table: pd.DataFrame) -> str:
    values = []
    for _, row in table.iterrows():
        values.extend(normalize_text(value) for value in row.tolist())
    return re.sub(r"\s+", " ", " ".join(value for value in values if value)).strip()


def make_race_id(race_no: int) -> str:
    return f"{DEFAULT_RACE_DATE_ID}{race_no:02d}"


def parse_3rentan(table: pd.DataFrame) -> tuple[str | None, int | None, int | None]:
    text = table_to_text(table)

    # 払戻表の下段は「単 5-2-1 1,950円 (2)」のように並びます。
    # 3つの車番を持つ「単」を三連単として扱います。
    matches = re.findall(r"単\s+(\d+-\d+-\d+)\s+([\d,]+)円\s*\((\d+)\)", text)
    if not matches:
        return None, None, None

    combination, payout_text, popularity_text = matches[-1]
    payout = int(payout_text.replace(",", ""))
    popularity = int(popularity_text)
    return combination, payout, popularity


def main() -> None:
    if not INPUT_HTML.exists():
        raise FileNotFoundError(f"{INPUT_HTML} が見つかりません。先に probe_kdreams.py を実行してください。")

    html = INPUT_HTML.read_text(encoding="utf-8")
    tables = pd.read_html(StringIO(html))

    rows = []
    race_no = 1
    for table_index in range(1, len(tables), 2):
        combination, payout, popularity = parse_3rentan(tables[table_index])
        if combination is None:
            continue

        rows.append(
            {
                "race_id": make_race_id(race_no),
                "bet_type": "3連単",
                "combination": combination,
                "payout": payout,
                "popularity": popularity,
            }
        )
        race_no += 1

    output = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"payouts CSVを作成しました: {OUTPUT_CSV}")
    print(output)


if __name__ == "__main__":
    main()
