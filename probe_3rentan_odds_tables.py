from io import StringIO
from pathlib import Path
import pandas as pd

HTML = Path(r"data\raw_kdreams_days\aomori\12202306050100\odds\1220230605010001.html")
OUT = Path(r"data\raw_probe\odds_3rentan_probe.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

html = HTML.read_text(encoding="utf-8", errors="ignore")

start = html.find("JS_ODDSCONTENTS_3rentan")
end = html.find("JS_ODDSCONTENTS_", start + 1)
block = html[start:end] if start != -1 and end != -1 else html[start:]

lines = []
lines.append(f"html: {HTML}")
lines.append(f"html length: {len(html)}")
lines.append(f"3rentan block start: {start}")
lines.append(f"3rentan block length: {len(block)}")
lines.append("")

for key in ["JS_ODDSCONTENTS_3rentan", "3連単", "人気順", "オッズ", "票数"]:
    lines.append(f"{key}: {html.find(key)}")

lines.append("\n=== read_html: 3rentan block ===")

try:
    tables = pd.read_html(StringIO(block))
    lines.append(f"tables: {len(tables)}")

    for i, table in enumerate(tables):
        lines.append(f"\n--- table {i} shape={table.shape} ---")
        lines.append(str(table.head(30)))
except Exception as e:
    lines.append(f"read_html error: {type(e).__name__}: {e}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"saved: {OUT}")
print("\n".join(lines[:20]))
