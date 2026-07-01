from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path("data") / "race.db"
TABLES = ["races", "entries", "payouts", "racers", "results", "odds", "lines"]


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} が見つかりません。先に make_db.py を実行してください。")

    with sqlite3.connect(DB_PATH) as conn:
        print("拡張DBの確認:")
        for table in TABLES:
            try:
                count = pd.read_sql_query(f"SELECT COUNT(*) AS count FROM {table}", conn)
                print(f"- {table}: {count.loc[0, 'count']} rows")
            except Exception as exc:
                print(f"- {table}: 未作成 ({exc})")

        print("\n荒れレース候補の確認:")
        query = """
        SELECT
            r.race_id,
            r.date,
            r.place,
            r.race_no,
            p.payout AS payout_3rentan,
            p.popularity AS payout_popularity
        FROM races r
        JOIN payouts p ON r.race_id = p.race_id
        WHERE p.bet_type = '3連単'
        ORDER BY p.payout DESC
        """
        print(pd.read_sql_query(query, conn))


if __name__ == "__main__":
    main()
