from pathlib import Path
import sqlite3

import pandas as pd


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "race.db"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。先に make_sample_data.py を実行してください。")
    return pd.read_csv(path)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    tables = {
        "races": load_csv("races"),
        "entries": load_csv("entries"),
        "payouts": load_csv("payouts"),
    }

    with sqlite3.connect(DB_PATH) as conn:
        for table_name, df in tables.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"SQLite DBを作成しました: {DB_PATH}")
    for table_name, df in tables.items():
        print(f"- {table_name}: {len(df)} rows")


if __name__ == "__main__":
    main()
