# 本物データ取得とDB拡張設計

## 方針

このプロジェクトは「勝者を当てるAI」ではなく、「三連単が高配当になりそうな荒れレース」を見つけるAIです。

最初の目的変数は `is_over_100` がおすすめです。30万円以上や50万円以上は発生数が少なく、データが少ない段階では学習が不安定になりやすいためです。

## 拡張テーブル

- `racers`: 選手の基本情報
- `results`: 着順や決まり手。レース後情報なので、予想時の特徴量には使いません。
- `odds`: 三連単オッズと人気順。予想直前に取れるなら強い特徴量です。
- `lines`: 並び、ライン番号、ライン内位置。競輪の荒れ予想では重要です。

## 本物データで最低限ほしいもの

- `races`: 開催日、場、レース番号、グレード、天候、風速
- `entries`: 車番、選手名、競走得点、脚質、年齢、勝率、連対率/3連対率
- `payouts`: 三連単の組み合わせ、配当、人気
- `odds`: 三連単オッズ、人気順
- `lines`: 並び、ライン番号、ライン内位置
- `results`: 着順、決まり手

## 取得コードの作り方

本物サイトはHTML構造が変わることがあるため、最初から巨大なスクレイパーにしない方が安全です。

おすすめ順序:

1. 1日分のレースID一覧を取る
2. 出走表ページを `data/raw/` に保存する
3. 結果ページを `data/raw/` に保存する
4. 保存HTMLからCSVへ変換する
5. CSVをSQLiteへ入れる
6. `features.csv` を作る

## 実行順

基本サンプル:

```bash
python make_sample_data.py
python make_db.py
python check_db.py
python make_features.py
```

拡張サンプル:

```bash
python make_extended_sample_data.py
python migrate_db.py
python check_extended_db.py
```

## 次の実装候補

- `fetch_race_pages.py`: 過去レースページを `data/raw/` に保存
- `parse_entries.py`: 出走表HTMLから `entries.csv` を作る
- `parse_results.py`: 結果HTMLから `payouts.csv` / `results.csv` を作る
- `parse_odds.py`: オッズHTMLから `odds.csv` を作る
- `make_features_v2.py`: `odds` / `lines` を含めた特徴量を作る
- `train_model.py`: 荒れ判定モデルを学習する
- `predict_today.py`: 今日のレースで荒れ候補を出す
