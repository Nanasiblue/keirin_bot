# 競輪AIプロジェクト

目的は、通常の勝者予想ではなく「三連単が高配当になりそうな荒れレース」を判定するAIを作ることです。

この段階では、本物の過去データ取得や学習モデル作成の前に、CSVからSQLiteを作り、AI学習用の`features.csv`を作る流れを確認します。

## 構成

```text
data/
  races.csv
  entries.csv
  payouts.csv
  race.db
  features.csv

make_sample_data.py
make_db.py
check_db.py
make_features.py
README.md
```

## 実行順

```bash
python make_sample_data.py
python make_db.py
python check_db.py
python make_features.py
```

## 入力CSV

### races.csv

```text
race_id,date,place,race_no,grade,weather,wind_speed
```

### entries.csv

```text
race_id,car_no,name,score,style,age,win_rate,place_rate
```

### payouts.csv

```text
race_id,bet_type,combination,payout,popularity
```

## features.csv の特徴量

レース単位で以下の特徴量を作ります。

```text
avg_score
max_score
min_score
std_score
score_gap
avg_age
racer_count
avg_win_rate
max_win_rate
avg_place_rate
max_place_rate
nige_count
oikomi_count
ryo_count
sashi_count
makuri_count
front_runner_pressure
is_rain
is_strong_wind
payout_3rentan
payout_popularity
is_over_100
is_over_300
is_over_500
```

`is_over_100`、`is_over_300`、`is_over_500`は、それぞれ三連単配当が10万円、30万円、50万円以上かどうかを表す目的変数候補です。

## 次に拡張したいこと

- 本物の過去レースデータ取得
- 選手、ライン、レース条件、オッズ、人気順のDB設計拡張
- 荒れレース判定モデルの学習
- `is_over_100`などを目的変数にした分類モデル作成

## 拡張DB

本物データ取得に向けて、以下のテーブルを追加できます。

- `racers`: 選手基本情報
- `results`: 着順、決まり手
- `odds`: 三連単オッズ、人気順
- `lines`: 並び、ライン情報

拡張サンプルの実行順:

```bash
python make_extended_sample_data.py
python migrate_db.py
python check_extended_db.py
```

詳しい設計は `DATA_DESIGN.md` を見てください。

## Kドリームス 1日分取得

1場1日分の実データを取得して、日別CSVと集計CSVを作ります。

```bash
python fetch_kdreams_day.py --place toride --race-date-id 23202606270100
```

作成される主なファイル:

```text
data/kdreams_days/{place}/{race_date_id}/entries.csv
data/kdreams_days/{place}/{race_date_id}/payouts.csv
data/kdreams_days/{place}/{race_date_id}/features.csv
data/entries_all_kdreams.csv
data/payouts_all_kdreams.csv
data/features_all_kdreams.csv
```

既に保存済みのHTMLは再利用します。取り直したい場合は `--force` を付けます。


## Kドリームス 複数日取得

日付リストCSVを順番に処理して、複数日分を集計CSVへ貯めます。

```bash
python fetch_kdreams_many_days.py --limit 1
python fetch_kdreams_many_days.py
```

日付リスト:

```text
data/kdreams_days_list.csv
```

ログ:

```text
data/fetch_logs_kdreams.csv
```

まずは `--limit 1` で1日だけ試してから、問題なければ制限なしで回してください。


## 取得失敗時の再実行

長期間取得では、途中失敗や未確定ページがありえます。以下のオプションを使えます。

```bash
python fetch_kdreams_day.py --place toride --race-date-id 23202606270100 --retries 3 --retry-wait 5
python fetch_kdreams_many_days.py --skip-success
python fetch_kdreams_many_days.py --only-failed --force
```

- `--retries`: 取得失敗時の再試行回数
- `--retry-wait`: 再試行前の待機秒数。回数ごとに少し長く待ちます
- `--skip-success`: 既にfeatures.csvがある日をスキップ
- `--only-failed`: 最新ログがerrorの日だけ再実行
- `--force`: 保存済みHTMLを使わず取り直し

HTMLが短すぎる、エラーページ、結果未確定、entries/payouts/featuresが0件の場合は失敗扱いになります。


## Kドリームス 開催日リスト作成

開催ページから、取得対象の `place,race_date_id,race_count` をCSVにします。

```bash
python make_kdreams_days_list.py --date 2026-06-27
python fetch_kdreams_many_days.py --days-file data/kdreams_days_list_generated.csv --skip-success
```

範囲指定もできます。最初は1週間くらいで試してください。

```bash
python make_kdreams_days_list.py --start 2026-06-01 --end 2026-06-07
python fetch_kdreams_many_days.py --days-file data/kdreams_days_list_generated.csv --skip-success
```


## Kドリームス 長期間取得

2023年1月1日から2026年6月26日までのリストを作る場合:

```bash
python make_kdreams_days_list.py --start 2023-01-01 --end 2026-06-26 --output data/kdreams_days_list_2023_20260626.csv --replace --sleep 0.5
```

作ったリストを使って、成功済みを飛ばしながら取得します。

```bash
python fetch_kdreams_many_days.py --days-file data/kdreams_days_list_2023_20260626.csv --skip-success --sleep 1.0 --retries 2 --retry-wait 5
```

途中で止まった場合も、同じコマンドをもう一度実行すれば続きから進みます。
