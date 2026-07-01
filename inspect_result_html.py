from pathlib import Path
import re


INPUT_HTML = Path("data") / "raw_probe" / "result.html"
OUTPUT_TXT = Path("data") / "raw_probe" / "result_snippets.txt"
KEYWORDS = ["3連単", "三連単", "払戻", "払戻金", "人気", "着順", "確定", "result"]


def clean_html(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def collect_snippets(html: str, keyword: str, width: int = 1200) -> list[str]:
    snippets = []
    start = 0
    while len(snippets) < 8:
        index = html.find(keyword, start)
        if index < 0:
            break
        left = max(0, index - width)
        right = min(len(html), index + width)
        snippets.append(clean_html(html[left:right]))
        start = index + len(keyword)
    return snippets


def main() -> None:
    if not INPUT_HTML.exists():
        raise FileNotFoundError(f"{INPUT_HTML} が見つかりません。先に probe_kdreams.py を実行してください。")

    html = INPUT_HTML.read_text(encoding="utf-8")
    lines = [f"source: {INPUT_HTML}", f"html length: {len(html)}", ""]

    for keyword in KEYWORDS:
        snippets = collect_snippets(html, keyword)
        lines.append(f"=== {keyword} ({len(snippets)} hits shown) ===")
        if not snippets:
            lines.append("not found")
        for i, snippet in enumerate(snippets, start=1):
            lines.append(f"--- snippet {i} ---")
            lines.append(snippet)
        lines.append("")

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(f"結果HTMLの確認用テキストを作成しました: {OUTPUT_TXT}")
    print("まずは result_snippets.txt の 3連単 / 払戻金 周辺を見て、payouts の構造を確認します。")


if __name__ == "__main__":
    main()
