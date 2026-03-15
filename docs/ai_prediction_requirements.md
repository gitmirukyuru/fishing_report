# AI釣果予測 要件定義書

## 1. 概要

### 目的
過去の釣果CSV・気象・潮汐データを学習し、指定日の釣行条件から
「釣果数・釣れやすい魚種・磯の期待度・釣行推奨スコア」をAIで予測する。

### アーキテクチャ方針
```
[特徴量エンジニアリング]
        ↓
[LightGBM（数値予測）]  →  釣果数予測・磯スコア・魚種確率
        ↓
[プロンプトテキスト生成] →  ユーザーが手動でClaude.aiに貼り付け
```

> **注**: Claude API（有償）は当面使用しない。
> ダッシュボードで「Claudeへのプロンプトをコピー」ボタンを提供し、
> ユーザーが claude.ai に手動で貼り付けてアドバイスを得る。

---

## 2. 入力特徴量

### 2-1. 気象系
| 特徴量 | 取得元 | 備考 |
|---|---|---|
| 天気 | tenki.jp / Open-Meteo | カテゴリ |
| 最高気温 / 最低気温 | tenki.jp / Open-Meteo | 数値 |
| 風速 (m/s) | Open-Meteo hourly | 釣行時間帯（6-18時）の平均 |
| 風向 | Open-Meteo hourly | 16方位カテゴリ |
| 波高 (m) | 釣果CSV `wave_height_m` | 数値 |
| 前日雨量 (mm) | Open-Meteo Archive | `precipitation_sum` |
| 2日前雨量 (mm) | Open-Meteo Archive | 同上 |
| 3日前雨量 (mm) | Open-Meteo Archive | 同上 |

### 2-2. 潮汐系
| 特徴量 | 取得元 | 備考 |
|---|---|---|
| 潮名 | tide736.net | 大潮/中潮/小潮/若潮/長潮 |
| 満潮・干潮時刻 | tide736.net | 各日最大4点 |
| 7〜14時の上り潮割合 | tide736.net から算出 | 0.0〜1.0。上り潮の時間 ÷ 7時間 |
| 7〜14時の下り潮割合 | 上記の補数 | 1 − 上り潮割合 |

> **上り潮割合の算出方法**
> 7〜14時の間で「直前の潮位変化が正方向（上昇）」の時間帯の割合。
> 満潮・干潮時刻から線形補間して各時刻の潮位を推定し判定する。

### 2-3. 時期系
| 特徴量 | 取得元 | 備考 |
|---|---|---|
| 月 | date から生成 | 1〜12 |
| 曜日 | date から生成 | 0(月)〜6(日) |
| 季節 | 月から生成 | 春/夏/秋/冬 |

### 2-4. 場所・魚種系
| 特徴量 | 取得元 | 備考 |
|---|---|---|
| 釣り場 (spot) | 釣果CSV | カテゴリ（Label Encoding） |
| 魚種 (species) | 釣果CSV | モデルによっては魚種別に分割 |

### 2-5. 水温
| 特徴量 | 取得元 | 備考 |
|---|---|---|
| 水温 (℃) | 釣果CSV `water_temp_avg` | 予測時は直近7日の移動平均で代替 |

---

## 3. 出力（予測ターゲット）

| 出力 | モデル種別 | 説明 |
|---|---|---|
| **期待釣果数**（一人あたり匹数） | 回帰 | LightGBM Regressor |
| **釣れやすい魚種ランキング** | 多クラス分類 | 各魚種の出現確率上位3種 |
| **釣行推奨スコア** | 二値分類 | 「釣れる日」確率（count≥5を正例） |
| **磯の期待度ランキング** | スコアリング | スポット別の予測count平均でソート |
| **Claudeへのプロンプト** | テキスト生成 | 予測結果をまとめたプロンプト文をコピーボタンで提供。ユーザーが claude.ai に手動で貼り付ける |

---

## 4. モデル構成

### 4-1. LightGBMモデル群

```
model_A: 釣果数回帰（全魚種合計count / 人）
  特徴量: 全2-1〜2-5
  ターゲット: count（1レコード = 1人分）

model_B: 釣行推奨分類（Go/No-Go）
  特徴量: 全2-1〜2-4（水温は欠損多いため除外も検討）
  ターゲット: count >= 5 を 1、それ以外を 0

model_C: 魚種別出現確率（魚種ごとの二値分類 × N魚種）
  ターゲット: その日・その磯でその魚種が釣れたか
```

### 4-2. Claudeへのプロンプト生成（手動貼り付け方式）

ダッシュボードで以下のテキストを生成し、コピーボタンで提供する。
ユーザーが claude.ai に貼り付けてアドバイスを取得する。

**プロンプトテンプレート**
```
以下の条件での磯釣り釣行についてアドバイスをください。

【釣行予定日】{date}
【気象】天気: {weather} / 気温: {temp_min}〜{temp_max}℃ / 風: {wind_dir} {wind_speed}m/s
【海況】波高: {wave}m / 水温: {water_temp}℃
【潮汐】潮名: {tide_name} / 上り潮割合（7-14時）: {rising_ratio}%
【雨歴】前日: {precip_1d}mm / 2日前: {precip_2d}mm / 3日前: {precip_3d}mm

【AI予測結果】
- 期待釣果: {predicted_count}匹（一人あたり）
- 釣行推奨スコア: {score}%
- 釣れやすい魚種: {species_rank}
- 過去類似条件: {n_similar}件中平均{avg_count}匹

推奨する仕掛け・エサ・狙い目の時間帯を含めて200字程度でアドバイスをください。
```

---

## 5. 追加データ取得が必要な項目

現在のCSVに存在しない特徴量は、Open-Meteo Archive APIを使って
過去日付分を一括取得・補完する。

| 項目 | API | パラメータ |
|---|---|---|
| 日別雨量 | `archive-api.open-meteo.com` | `daily=precipitation_sum` |
| 風速最大値 | 同上 | `daily=windspeed_10m_max` |
| 風向（卓越） | 同上 | `daily=winddirection_10m_dominant` |
| 過去潮汐 | `api.tide736.net` | `rg=day` 日別（学習データ分） |

---

## 6. 新規ファイル構成

```
fishing/
├── ml/
│   ├── feature_builder.py   # 特徴量エンジニアリング（雨量・潮汐・風向計算）
│   ├── train.py             # LightGBMモデル学習・保存（models/*.pkl）
│   ├── predict.py           # 予測API（dashboard.py から呼び出し）
│   └── models/              # 学習済みモデル（.pkl）
├── prompt_builder.py        # Claudeへの手動貼り付け用プロンプト文を生成
└── docs/
    └── ai_prediction_requirements.md  # 本ファイル
```

---

## 7. ダッシュボード変更方針

### 「🎣 釣果予測」タブの構成変更

```
[現在]
  - 折れ線グラフ（気温・潮汐）
  - 期待釣果スコアテーブル（ルールベース: 波高±0.5m・水温±1℃）
  - 日別エキスパンダー

[変更後]
  - 折れ線グラフ（気温・潮汐） ← 維持
  - AI予測セクション（新規）
      ├── 釣行推奨スコアゲージ（0〜100%）
      ├── 期待釣果数（匹）
      ├── 釣れやすい魚種TOP3
      ├── 磯の期待度ランキング（上位10磯）
      └── Claude AIによる釣行アドバイス文
  - 日別エキスパンダー ← 維持
```

---

## 8. 実装フェーズ計画

| フェーズ | 内容 |
|---|---|
| **Phase 1** | `feature_builder.py`: 過去データへの雨量・風・潮汐特徴量の付加 |
| **Phase 2** | `train.py`: LightGBM 3モデルの学習・評価・保存 |
| **Phase 3** | `predict.py`: 予測日の特徴量生成 → モデル推論 |
| **Phase 4** | `prompt_builder.py`: Claudeへの手動貼り付け用プロンプト生成 |
| **Phase 5** | `dashboard.py`: Tab1にAI予測UI + プロンプトコピーボタンを統合 |

---

## 9. 決定事項・未決定事項

### 決定済み
- [x] **モデル再学習**: スクレイパー実行後（`scraper.py` 終了時）に自動で `train.py` を呼び出す
  - `scraper.py` の末尾で `subprocess.run(["python", "ml/train.py"])` を実行
  - 新規データが0件のときは再学習をスキップ
- [x] **Claude APIキー管理**: `.env` ファイル + `python-dotenv` によるベストプラクティス運用
  （詳細は下記「10. Claude APIキー管理方針」参照）

- [x] **水温の欠損値補完**: Open-Meteo Marine API の `sea_surface_temperature` で補完（案C）
  - 学習・予測の両フェーズで一貫して使用
- [x] **磯の扱い**: 10件以上の磯（29磯）を個別カテゴリ、それ未満（78磯）を「その他」に集約
- ~~LLMアドバイスのキャッシュ方針~~ → Claude API不使用により削除

### 未決定
なし。全項目決定済み。

---

## 10. Claude APIキー管理方針

### ベストプラクティス: `.env` ファイル + `python-dotenv`

**ファイル構成**
```
fishing/
├── .env          ← APIキーを記載（絶対にGitにコミットしない）
├── .gitignore    ← .env を除外
└── advisor.py    ← os.environ 経由でキーを参照
```

**`.env` の内容**
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

**`.gitignore` に追加する行**
```
.env
ml/models/
```

**Python側の読み込み（`advisor.py`）**
```python
from dotenv import load_dotenv
import os

load_dotenv()  # .env を自動読み込み
api_key = os.environ["ANTHROPIC_API_KEY"]  # 未設定なら KeyError で即失敗
```

**なぜこの方式か**
| 方法 | 問題点 |
|---|---|
| コードにベタ書き | Git履歴に残る。絶対NG |
| 環境変数をOSに直設定 | 他ユーザーに見える可能性。再起動で消える場合も |
| `.env` + dotenv | ローカル限定、Git除外、コード変更不要でキー差し替え可能 |

**インストール**
```bash
pip install python-dotenv anthropic
```
`requirements.txt` に追記:
```
python-dotenv>=1.0
anthropic>=0.25
```
