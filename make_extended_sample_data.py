from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    racers = pd.DataFrame([
        {"racer_id": "R001", "name": "佐藤 一郎", "prefecture": "東京", "rank": "A1", "birth_year": 1992},
        {"racer_id": "R002", "name": "鈴木 次郎", "prefecture": "埼玉", "rank": "A2", "birth_year": 1999},
        {"racer_id": "R003", "name": "高橋 三郎", "prefecture": "神奈川", "rank": "A1", "birth_year": 1995},
        {"racer_id": "R004", "name": "田中 四郎", "prefecture": "千葉", "rank": "A2", "birth_year": 1987},
        {"racer_id": "R005", "name": "伊藤 五郎", "prefecture": "静岡", "rank": "A2", "birth_year": 1990},
        {"racer_id": "R006", "name": "山本 六郎", "prefecture": "大阪", "rank": "S2", "birth_year": 2001},
        {"racer_id": "R007", "name": "中村 七郎", "prefecture": "京都", "rank": "A2", "birth_year": 1984},
        {"racer_id": "R008", "name": "小林 八郎", "prefecture": "兵庫", "rank": "A2", "birth_year": 1981},
        {"racer_id": "R009", "name": "加藤 九郎", "prefecture": "奈良", "rank": "A1", "birth_year": 1993},
        {"racer_id": "R010", "name": "吉田 十郎", "prefecture": "滋賀", "rank": "A2", "birth_year": 1978},
        {"racer_id": "R011", "name": "渡辺 十一", "prefecture": "和歌山", "rank": "A2", "birth_year": 1989},
    ])

    results = pd.DataFrame([
        {"race_id": "202601010101", "car_no": 1, "finish_position": 1, "final_home_position": 2, "final_back_position": 2, "decision": "差し"},
        {"race_id": "202601010101", "car_no": 3, "finish_position": 2, "final_home_position": 4, "final_back_position": 3, "decision": "捲り"},
        {"race_id": "202601010101", "car_no": 2, "finish_position": 3, "final_home_position": 1, "final_back_position": 1, "decision": "逃げ"},
        {"race_id": "202601010102", "car_no": 4, "finish_position": 1, "final_home_position": 5, "final_back_position": 4, "decision": "捲り"},
        {"race_id": "202601010102", "car_no": 2, "finish_position": 2, "final_home_position": 3, "final_back_position": 3, "decision": "追込"},
        {"race_id": "202601010102", "car_no": 5, "finish_position": 3, "final_home_position": 6, "final_back_position": 6, "decision": "差し"},
    ])

    odds = pd.DataFrame([
        {"race_id": "202601010101", "bet_type": "3連単", "combination": "1-3-2", "odds": 84.5, "popularity": 4},
        {"race_id": "202601010101", "bet_type": "3連単", "combination": "1-2-3", "odds": 12.8, "popularity": 1},
        {"race_id": "202601010101", "bet_type": "3連単", "combination": "3-1-2", "odds": 46.2, "popularity": 2},
        {"race_id": "202601010102", "bet_type": "3連単", "combination": "4-2-5", "odds": 3825.0, "popularity": 218},
        {"race_id": "202601010102", "bet_type": "3連単", "combination": "1-4-2", "odds": 18.4, "popularity": 1},
        {"race_id": "202601010102", "bet_type": "3連単", "combination": "1-2-4", "odds": 24.9, "popularity": 2},
    ])

    lines = pd.DataFrame([
        {"race_id": "202601010101", "line_no": 1, "car_no": 2, "line_position": 1},
        {"race_id": "202601010101", "line_no": 1, "car_no": 1, "line_position": 2},
        {"race_id": "202601010101", "line_no": 2, "car_no": 3, "line_position": 1},
        {"race_id": "202601010101", "line_no": 2, "car_no": 5, "line_position": 2},
        {"race_id": "202601010101", "line_no": 3, "car_no": 4, "line_position": 1},
        {"race_id": "202601010102", "line_no": 1, "car_no": 1, "line_position": 1},
        {"race_id": "202601010102", "line_no": 1, "car_no": 2, "line_position": 2},
        {"race_id": "202601010102", "line_no": 1, "car_no": 3, "line_position": 3},
        {"race_id": "202601010102", "line_no": 2, "car_no": 4, "line_position": 1},
        {"race_id": "202601010102", "line_no": 2, "car_no": 5, "line_position": 2},
        {"race_id": "202601010102", "line_no": 3, "car_no": 6, "line_position": 1},
    ])

    racers.to_csv(DATA_DIR / "racers.csv", index=False, encoding="utf-8-sig")
    results.to_csv(DATA_DIR / "results.csv", index=False, encoding="utf-8-sig")
    odds.to_csv(DATA_DIR / "odds.csv", index=False, encoding="utf-8-sig")
    lines.to_csv(DATA_DIR / "lines.csv", index=False, encoding="utf-8-sig")

    print("拡張サンプルCSVを作成しました。")
    print("- data/racers.csv")
    print("- data/results.csv")
    print("- data/odds.csv")
    print("- data/lines.csv")


if __name__ == "__main__":
    main()
