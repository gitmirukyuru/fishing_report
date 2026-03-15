# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要
谷口渡船の磯釣り釣果ページから釣果データを取得・構造化し、ダッシュボードで可視化するツール。
ライブラリ：Scrapling / pandas / Python 3.10+ / Streamlit / Plotly / Open-Meteo API（無料・APIキー不要）

## コマンド
```bash
pip install -r requirements.txt
python scraper.py              # インクリメンタル取得（state.json の日付以降）
python scraper.py --days 7     # 直近7日分取得
python scraper.py --full       # 全60ページ取得
streamlit run dashboard.py     # ダッシュボード起動
```

## ファイル構成
- `scraper.py`     — エントリポイント・ページネーション制御・state.json 管理
- `parser.py`      — HTML解析・データ抽出（1ページ単位）
- `storage.py`     — CSV書き込み・重複排除
- `weather_api.py` — Open-Meteo APIラッパー（予報・過去データ取得）
- `dashboard.py`   — Streamlitダッシュボード
- `state.json`     — 前回取得日付の記録 `{"last_date": "YYYY-MM-DD"}`
- `output/`        — 出力先（`YYYYMMDD_rockshore.csv`）

## 対象URL
- 一覧: `https://www.taniguchitosen.com/rockshore`
- ページネーション: `/rockshore/page/2` 〜 `/rockshore/page/60`
- アプリ公開用URL: `https://fishingreport-kh2suneznc8cnv2unyspma.streamlit.app/`

## データモデル（1レコード）
| フィールド | 内容 |
|---|---|
| date | ISO形式 YYYY-MM-DD |
| day_of_week | 曜日（月〜日） |
| weather | 天候テキスト |
| water_temp_raw | 水温の生テキスト |
| water_temp_avg | 水温の数値平均（float） |
| wave_height_raw | 波高の生テキスト |
| wave_height_m | 波高の数値（float, m）先頭の数値を抽出 |
| next_day_status | 翌日情報テキスト |
| spot | 釣り場名 |
| angler | 釣り人名 |
| species_raw | 魚種の生テキスト（例: 串本11匹32cm〜40.2cm） |
| species | 正規化済み魚種名 |
| count | 匹数（int） |
| size_min_cm / size_max_cm | サイズ範囲（float, cm） |
| tackle | 仕掛け |
| bait | エサ |
| detail_url | 詳細ページURL |

## アーキテクチャ

### パース方針
ページHTMLは `<h3 class="entry_title ...">` タグで日付ブロックを区切る構造（ラッパーdivなし）。
`re.split(r'(<h3[^>]*>.*?</h3>)', html)` でセクション分割し、各セクションを個別にパースする。

- **天候・水温・波高**: `<div class="post_data">` 内の `<p>` タグに `<br>` 区切りで記載
- **翌日情報**: 同 `<div>` 内の2番目以降の `<p>` タグに記載
- **detail_url**: 詳細ページありの日付は `<h3>` 内の `<a href>` に入っている（`&amp;` エンティティに注意）

**テーブル構造（2列）**:
```html

  釣り場釣り人魚種仕掛けエサ
  沖の長島藤井様グレ8匹30cm〜40.5cmフカセオキアミ

```
魚種セル例: `串本11匹32cm〜40.2cm`（魚種名＋匹数＋サイズが1セルに結合）

### 状態管理
`state.json` に最終取得日を保存。次回起動時にその日付以降のみ取得（インクリメンタル）。
`--full` フラグはカットオフを無視して全ページ取得する。

### 重複排除キー
`date + spot + angler` の組み合わせ（storage.py の `DEDUP_KEYS`）

### 魚種正規化マッピング
`parser.py` の `SPECIES_MAP` に定義。マッピング外は `species_raw` をそのまま `species` に使用。
- `イサギ` → `イサキ`
- `串本` → `グレ`
- `グレ`が含まれていればすべて'グレ'とみなす

### レート制限
リクエスト間隔：1〜3秒ランダムウェイト（`time.sleep(random.uniform(1, 3))`）

## 天気予報API（weather_api.py）
Open-Meteo（https://open-meteo.com/）を使用。APIキー不要・商用無料。

**観測地点**
- 緯度: 33.4833 / 経度: 135.7833（串本町出雲崎付近）

**取得エンドポイント**
```
# 7日間予報
GET https://api.open-meteo.com/v1/forecast
  ?latitude=33.4833&longitude=135.7833
  &daily=weathercode,wave_height_max,temperature_2m_max
  &timezone=Asia/Tokyo

# 過去データ（釣果との照合用）
GET https://archive-api.open-meteo.com/v1/archive
  ?latitude=33.4833&longitude=135.7833
  &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
  &daily=weathercode,wave_height_max
  &timezone=Asia/Tokyo
```

**返却モデル**
```python
{
  "date": "2026-03-12",
  "forecast_weather_code": 1,       # WMO天気コード
  "forecast_wave_height_m": 1.5,
  "forecast_temp_max": 16.2
}
```

## ダッシュボード仕様（dashboard.py）

**フレームワーク**: Streamlit + Plotly Express

### レイアウト構成
```
[サイドバー]
  - 期間フィルタ（日付範囲）
  - 魚種フィルタ（multiselect）
  - 「最新データを取得」ボタン
    → scraper.py をサブプロセスで実行
    → 進捗をst.statusで表示

[メインエリア]
  タブ1: 釣行予測
    - 向こう7日間の天気・波高予報テーブル
    - 過去の同条件（波高±0.5m・水温±1℃）の釣果平均を「期待釣果スコア」として表示

  タブ2: 水温×釣果
    - X軸: water_temp_avg / Y軸: count / 色: species
    - 散布図（Plotly Express scatter）
    - 水温帯（1℃刻み）ごとの平均サイズ棒グラフ

  タブ3: 釣り場ランキング
    - 釣り場別の総釣果数・平均サイズ・最大サイズ・出現魚種一覧
    - 横棒グラフ（上位20磯）

  タブ4: 月別・季節別
    - 月×魚種のヒートマップ（釣果数）
    - 季節（春夏秋冬）ごとの魚種構成円グラフ

  タブ5: 波高・天候条件
    - 波高帯別の釣果数箱ひげ図
    - 天候×波高×水温の3変数バブルチャート
```

### データ取得ボタンの実装方針
```python
import subprocess
if st.sidebar.button("最新データを取得"):
    with st.status("スクレイピング中..."):
        result = subprocess.run(
            ["python", "scraper.py"],
            capture_output=True, text=True
        )
        st.write(result.stdout)
    st.rerun()
```

### キャッシュ方針
- `@st.cache_data(ttl=300)` をCSV読み込み関数に付与
- データ取得ボタン押下後は `st.cache_data.clear()` を呼び出す
```