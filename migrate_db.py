from pathlib import Path
import sqlite3

import pandas as pd


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "race.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS racers (
    racer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prefecture TEXT,
    rank TEXT,
    birth_year INTEGER
);

CREATE TABLE IF NOT EXISTS results (
    race_id TEXT NOT NULL,
    car_no INTEGER NOT NULL,
    finish_position INTEGER,
    final_home_position INTEGER,
    final_back_position INTEGER,
    decision TEXT,
    PRIMARY KEY (race_id, car_no)
);

CREATE TABLE IF NOT EXISTS odds (
    race_id TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    odds REAL,
    popularity INTEGER,
    PRIMARY KEY (race_id, bet_type, combination)
);

CREATE TABLE IF NOT EXISTS lines (
    race_id TEXT NOT NULL,
    line_no INTEGER NOT NULL,
    car_no INTEGER NOT NULL,
    line_position INTEGER,
    PRIMARY KEY (race_id, line_no, car_no)
);

CREATE INDEX IF NOT EXISTS idx_entries_race_id ON entries (race_id);
CREATE INDEX IF NOT EXISTS idx_payouts_race_id ON payouts (race_id);
CREATE INDEX IF NOT EXISTS idx_results_race_id ON results (race_id);
CREATE INDEX IF NOT EXISTS idx_odds_race_id ON odds (race_id);
CREATE INDEX IF NOT EXISTS idx_lines_race_id ON lines (race_id);
"""


OPTIONAL_TABLES = {
    "racers": "racers.csv",
    "results": "results.csv",
    "odds": "odds.csv",
    "lines": "lines.csv",
}


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} が見つかりません。先に make_db.py を実行してください。")

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA_SQL)

        for table_name, csv_name in OPTIONAL_TABLES.items():
            path = DATA_DIR / csv_name
            if path.exists():
                df = pd.read_csv(path)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"{table_name} を取り込みました: {len(df)} rows")
            else:
                print(f"{path} がないため {table_name} は空のままです。")

    print(f"DB拡張が完了しました: {DB_PATH}")


if __name__ == "__main__":
    main()
