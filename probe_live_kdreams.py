from __future__ import annotations

import re
import time
import html
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd


JST = timezone(timedelta(hours=9))
BASE = "https://keirin.kdreams.jp"
RACECARD_URL = "https://keirin.kdreams.jp/racecard/"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)

OUT_DIR = Path("data/live_input")
RAW_DIR = Path("data/live_raw")


@dataclass
class PageData:
    url: str
    title: str
    text: str
    links: list[str]


class SimpleHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self._in_title = False
        self.text_parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        self.text_parts.append(data)


def clean_text(s: str) -> str:
    s = html.unescape(str(s or ""))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fetch(url: str, timeout: int = 15, retries: int = 2) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as res:
                charset = res.headers.get_content_charset() or "utf-8"
                return res.read().decode(charset, errors="replace")
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            time.sleep(0.8 + i * 0.8)
    raise last_err


def parse_page(url: str, src: str) -> PageData:
    p = SimpleHTMLParser()
    p.feed(src)
    return PageData(
        url=url,
        title=clean_text(p.title),
        text=clean_text(" ".join(p.text_parts)),
        links=p.links,
    )


def normalize_url(url: str) -> str:
    url = urljoin(BASE, url)
    parsed = urlparse(url)
    return parsed.scheme + "://" + parsed.netloc + parsed.path


def extract_deadline_dt(text: str) -> datetime | None:
    date_m = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    time_m = re.search(r"投票締切\s*([0-2]?\d):([0-5]\d)", text)
    if not time_m:
        time_m = re.search(r"発走予定\s*([0-2]?\d):([0-5]\d)", text)

    if not date_m or not time_m:
        return None

    y, mo, d = map(int, date_m.groups())
    h, mi = map(int, time_m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=JST)


def race_label_from_title(title: str) -> str:
    title = clean_text(title)
    m = re.search(r"(.+?競輪).*?(\d+R)", title)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return title[:80]


def extract_race_id(url: str) -> str:
    m = re.search(r"/racedetail/(\d+)/?", url)
    return m.group(1) if m else ""


def extract_race_urls(max_scan: int = 300) -> list[str]:
    src = fetch(RACECARD_URL)
    page = parse_page(RACECARD_URL, src)

    urls = []
    for href in page.links:
        if "/racedetail/" in href:
            urls.append(normalize_url(href))

    urls = sorted(set(urls))
    if max_scan:
        urls = urls[:max_scan]

    return urls


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(JST)
    target_date = now.strftime("%Y-%m-%d")
    print(f"now JST: {now:%Y-%m-%d %H:%M}")
    print("fetching racecard...")

    urls = extract_race_urls()
    print(f"race urls: {len(urls)}")

    rows = []

    for i, url in enumerate(urls, start=1):
        try:
            src = fetch(url, retries=1)
            page = parse_page(url, src)
            deadline = extract_deadline_dt(page.text)

            if not deadline:
                continue
            if deadline <= now:
                continue

            race_id = extract_race_id(url)
            odds_url = normalize_url(url) + "?kakeshikiType=3rentan&pageType=odds"

            # 軽くオッズページも確認
            odds_src = fetch(odds_url, retries=1)
            has_3rentan = "3連単" in odds_src
            has_odds = "オッズ" in odds_src

            raw_path = RAW_DIR / f"{race_id}_odds.html"
            raw_path.write_text(odds_src, encoding="utf-8")

            rows.append({
                "race_id": race_id,
                "label": race_label_from_title(page.title),
                "title": page.title,
                "deadline_jst": deadline.strftime("%Y-%m-%d %H:%M"),
                "url": normalize_url(url),
                "odds_url": odds_url,
                "has_3rentan": int(has_3rentan),
                "has_odds": int(has_odds),
                "raw_odds_html": str(raw_path),
            })

            print(f"[OK] {deadline:%H:%M} {race_label_from_title(page.title)} {race_id}")

        except Exception as e:
            print(f"[NG] {url} / {e}")

    df = pd.DataFrame(rows).sort_values("deadline_jst") if rows else pd.DataFrame()
    out = OUT_DIR / f"open_races_{target_date}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("")
    print(f"open races: {len(df)}")
    print(f"saved: {out}")

    if not df.empty:
        print(df[["deadline_jst", "label", "race_id", "has_3rentan", "has_odds"]].to_string(index=False))


if __name__ == "__main__":
    main()
