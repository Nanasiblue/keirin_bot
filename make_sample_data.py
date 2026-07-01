from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    races = pd.DataFrame(
        [
            {
                "race_id": "202601010101",
                "date": "2026-01-01",
                "place": "立川",
                "race_no": 1,
                "grade": "F1",
                "weather": "晴",
                "wind_speed": 1.2,
            },
            {
                "race_id": "202601010102",
                "date": "2026-01-01",
                "place": "立川",
                "race_no": 2,
                "grade": "F1",
                "weather": "雨",
                "wind_speed": 5.8,
            },
        ]
    )

    entries = pd.DataFrame(
        [
            {"race_id": "202601010101", "car_no": 1, "name": "佐藤 一郎", "score": 92.1, "style": "追込", "age": 34, "win_rate": 0.18, "place_rate": 0.45},
            {"race_id": "202601010101", "car_no": 2, "name": "鈴木 次郎", "score": 88.4, "style": "逃げ", "age": 27, "win_rate": 0.12, "place_rate": 0.38},
            {"race_id": "202601010101", "car_no": 3, "name": "高橋 三郎", "score": 90.3, "style": "捲り", "age": 31, "win_rate": 0.15, "place_rate": 0.41},
            {"race_id": "202601010101", "car_no": 4, "name": "田中 四郎", "score": 85.2, "style": "両", "age": 39, "win_rate": 0.08, "place_rate": 0.30},
            {"race_id": "202601010101", "car_no": 5, "name": "伊藤 五郎", "score": 87.9, "style": "差し", "age": 36, "win_rate": 0.10, "place_rate": 0.34},
            {"race_id": "202601010102", "car_no": 1, "name": "山本 六郎", "score": 96.7, "style": "逃げ", "age": 25, "win_rate": 0.31, "place_rate": 0.62},
            {"race_id": "202601010102", "car_no": 2, "name": "中村 七郎", "score": 83.5, "style": "追込", "age": 42, "win_rate": 0.05, "place_rate": 0.22},
            {"race_id": "202601010102", "car_no": 3, "name": "小林 八郎", "score": 82.8, "style": "追込", "age": 45, "win_rate": 0.04, "place_rate": 0.20},
            {"race_id": "202601010102", "car_no": 4, "name": "加藤 九郎", "score": 89.6, "style": "捲り", "age": 33, "win_rate": 0.14, "place_rate": 0.37},
            {"race_id": "202601010102", "car_no": 5, "name": "吉田 十郎", "score": 81.9, "style": "差し", "age": 48, "win_rate": 0.03, "place_rate": 0.18},
            {"race_id": "202601010102", "car_no": 6, "name": "渡辺 十一", "score": 86.2, "style": "両", "age": 37, "win_rate": 0.09, "place_rate": 0.28},
        ]
    )

    payouts = pd.DataFrame(
        [
            {"race_id": "202601010101", "bet_type": "3連単", "combination": "1-3-2", "payout": 8450, "popularity": 4},
            {"race_id": "202601010102", "bet_type": "3連単", "combination": "4-2-5", "payout": 382500, "popularity": 218},
        ]
    )

    races.to_csv(DATA_DIR / "races.csv", index=False, encoding="utf-8-sig")
    entries.to_csv(DATA_DIR / "entries.csv", index=False, encoding="utf-8-sig")
    payouts.to_csv(DATA_DIR / "payouts.csv", index=False, encoding="utf-8-sig")

    print("サンプルCSVを作成しました。")
    print(f"- {DATA_DIR / 'races.csv'}")
    print(f"- {DATA_DIR / 'entries.csv'}")
    print(f"- {DATA_DIR / 'payouts.csv'}")


if __name__ == "__main__":
    main()
