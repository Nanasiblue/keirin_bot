# Keirin AI Fixed Rules

## direct_v1_middle_odds

固定日: 2026-07-01

目的:
- 三連単の中穴帯を狙う
- 過去検証で良かったルールを固定し、6/28以降の未使用データで前向きテストする

条件:
- odds >= 30
- odds <= 500
- direct_ticket_score >= 0.9
- 各レース direct_ticket_score 上位2点まで

検証期間:
- 2026-01-01 から 2026-06-26

検証結果:
- race_count: 854
- tickets: 890
- ROI: 112.85%
- ROI without max hit: 109.04%
- hit_count: 33
- month_min_roi: 86.28%
- plus_months: 3 / 6

注意:
- これは本番確定ではなく、前向きテスト用の第一候補ルール。
- 6/28以降の未使用データで確認するまで、実資金投入はしない。
