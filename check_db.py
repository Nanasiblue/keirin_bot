from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path("data") / "race.db"


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} が見つかりません。先に make_db.py を実行してください。")

    with sqlite3.connect(DB_PATH) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
            conn,
        )

        print("SQLite内のテーブル:")
        for table_name in tables["name"]:
            count = pd.read_sql_query(f"SELECT COUNT(*) AS count FROM {table_name}", conn)
            print(f"- {table_name}: {count.loc[0, 'count']} rows")

        print("\nraces:")
        print(pd.read_sql_query("SELECT * FROM races ORDER BY race_id", conn))

        print("\nentries:")
        print(pd.read_sql_query("SELECT * FROM entries ORDER BY race_id, car_no", conn))

        print("\npayouts:")
        print(pd.read_sql_query("SELECT * FROM payouts ORDER BY race_id", conn))


if __name__ == "__main__":
    main()
