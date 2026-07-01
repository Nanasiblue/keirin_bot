from pathlib import Path
from urllib.request import Request, urlopen
from io import StringIO

import pandas as pd


RAW_DIR = Path("data") / "raw_probe"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 初日で統一しています。entries と payouts の race_id が結合できるようにするためです。
URLS = {
    "racecard": "https://keirin.kdreams.jp/toride/racecard/23202606270100/",
    "odds": "https://keirin.kdreams.jp/toride/racedetail/2320260627010001/?kakeshikiType=3rentan&pageType=odds",
    "result": "https://keirin.kdreams.jp/toride/raceresult/23202606270100/",
}

KEYWORDS = [
    "出走表", "選手名", "競走得点", "脚質", "オッズ", "3連単", "三連単",
    "払戻", "人気", "着順", "決まり手", "並び", "天候", "風速",
]


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


def show_tables(html: str) -> None:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        print("tables: 0")
        return
    except ImportError as exc:
        print(f"tables: skipped ({exc})")
        print("hint: python -m pip install lxml html5lib beautifulsoup4")
        return

    print(f"tables: {len(tables)}")
    for index, table in enumerate(tables[:5]):
        print(f"\n--- table {index} ---")
        print(table.head())


def main() -> None:
    for name, url in URLS.items():
        print(f"\n=== {name} ===")
        print(url)
        try:
            html = fetch(url)
        except Exception as exc:
            print(f"ERROR: 取得できませんでした: {exc}")
            continue
        output_path = RAW_DIR / f"{name}.html"
        output_path.write_text(html, encoding="utf-8")
        print(f"saved: {output_path}")
        print(f"html length: {len(html)}")
        found = [keyword for keyword in KEYWORDS if keyword in html]
        print("keywords:")
        for keyword in found:
            print(f"  OK: {keyword}")
        if not found:
            print("  見つかりませんでした")
        show_tables(html)


if __name__ == "__main__":
    main()
