from pathlib import Path
from urllib.request import Request, urlopen


RAW_DIR = Path("data") / "raw"


def download_html(url: str, output_name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / output_name

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; keirin-ai-study/0.1)",
        },
    )

    with urlopen(request, timeout=20) as response:
        html = response.read()

    output_path.write_bytes(html)
    return output_path


def main() -> None:
    print("本物データ取得の入口です。")
    print("まずは対象サイトのURLを1つ決めて、download_html(url, output_name) で data/raw/ に保存します。")
    print("例:")
    print("  from fetch_real_data import download_html")
    print("  download_html('https://example.com/race_page.html', 'sample_race.html')")


if __name__ == "__main__":
    main()
