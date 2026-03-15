"""
dashboard.py — 谷口渡船 釣果ダッシュボード
起動: streamlit run dashboard.py
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import weather_api
import tenki_scraper
import prompt_builder
try:
    import ml.predict as ml_predict
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

st.set_page_config(page_title='谷口渡船 釣果ダッシュボード', layout='wide', page_icon='🎣')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Hiragino Sans', 'Yu Gothic UI', sans-serif;
}

/* ── 背景 ── */
.stApp { background-color: #EDF2F7; }
.main .block-container { padding-top: 1.5rem; max-width: 1400px; }

/* ── サイドバー ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B3D5C 0%, #155A7E 50%, #0B3D5C 100%);
    border-right: none;
    box-shadow: 4px 0 20px rgba(0,0,0,0.15);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] span:not([data-baseweb]) {
    color: #B8D8EA !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

/* サイドバー日付・セレクト入力 */
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background-color: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.25) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: rgba(27,143,168,0.6) !important;
}

/* ── タブ ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 5px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    gap: 3px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 9px 18px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #4A7A95 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #EEF5FA !important;
    color: #0B3D5C !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1B8FA8, #0B3D5C) !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(27,143,168,0.4) !important;
}

/* ── Plotlyチャートカード ── */
[data-testid="stPlotlyChart"] > div {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 12px 16px 8px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07);
}

/* ── DataFrameカード ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}

/* ── Metricカード ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 4px solid #1B8FA8;
}

/* ── ボタン ── */
.stButton > button {
    background: linear-gradient(135deg, #1B8FA8 0%, #0B3D5C 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 18px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 10px rgba(11,61,92,0.25) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(11,61,92,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border-radius: 12px !important;
    border: 1px solid #D9E8F2 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
    margin-bottom: 8px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 500;
    color: #0B3D5C;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #F0F7FB !important;
}

/* ── Statusボックス ── */
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
}

/* ── Alertボックス ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── テキスト見出し ── */
h1 { color: #0B3D5C !important; }
h2, h3 { color: #0D4F72 !important; }
h5 { color: #1A6A8A !important; letter-spacing: 0.01em; }
hr { border-color: #D5E6EF; margin: 1.2rem 0; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #1B8FA8 !important; }

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background-color: #E0F0F7 !important;
    color: #0B3D5C !important;
    border-radius: 6px !important;
}

/* ── Caption ── */
.stCaptionContainer p { color: #5A8095 !important; }

/* ── メインエリア ドロップダウン（selectbox）── */
[data-baseweb="select"] > div:first-child {
    background-color: #FFFFFF !important;
    border: 1.5px solid #A8C8DC !important;
    border-radius: 8px !important;
    color: #0B3D5C !important;
}
[data-baseweb="select"] > div:first-child:hover {
    border-color: #1B8FA8 !important;
}
[data-baseweb="select"] svg { color: #1B8FA8 !important; }

/* ドロップダウンのメニューリスト */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #A8C8DC !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background-color: #EAF4FA !important;
}
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #D4EBF5 !important;
    color: #0B3D5C !important;
    font-weight: 600 !important;
}

/* ── テキスト入力・日付入力 ── */
[data-baseweb="input"] > div,
[data-baseweb="base-input"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #A8C8DC !important;
    border-radius: 8px !important;
    color: #0B3D5C !important;
}
[data-baseweb="input"] > div:focus-within,
[data-baseweb="base-input"]:focus-within {
    border-color: #1B8FA8 !important;
    box-shadow: 0 0 0 3px rgba(27,143,168,0.15) !important;
}

/* ── ラジオボタン ── */
[data-testid="stRadio"] > div {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 8px 14px;
    border: 1.5px solid #A8C8DC;
    display: inline-flex;
    gap: 8px;
}
[data-testid="stRadio"] label { color: #0B3D5C !important; font-weight: 500; }

/* ── マルチセレクト（メインエリア）── */
.stMultiSelect [data-baseweb="select"] > div:first-child {
    background-color: #FFFFFF !important;
    border: 1.5px solid #A8C8DC !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}

/* ── モバイル対応 ── */
@media (max-width: 640px) {
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 0.75rem !important;
    }
    /* ヘッダーを小さく */
    h1 { font-size: 1.2rem !important; }
    h2 { font-size: 1.05rem !important; }
    h3, h5 { font-size: 0.95rem !important; }
    /* タブラベルを詰める */
    .stTabs [data-baseweb="tab"] {
        padding: 7px 10px !important;
        font-size: 0.78rem !important;
    }
    /* Metricカードの余白縮小 */
    [data-testid="metric-container"] {
        padding: 10px 12px !important;
    }
    /* ボタン余白 */
    .stButton > button {
        padding: 8px 12px !important;
        font-size: 0.82rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

OUTPUT_DIR = Path('output')

# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """output/ 以下の全CSVを連結して返す（重複排除済み）。"""
    csvs = sorted(OUTPUT_DIR.glob('*_rockshore.csv'))
    if not csvs:
        return pd.DataFrame()
    dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset=['date', 'spot', 'angler', 'species'], keep='last', inplace=True)

    df['date']           = pd.to_datetime(df['date'], errors='coerce')
    df['count']          = pd.to_numeric(df['count'], errors='coerce')
    df['water_temp_avg'] = pd.to_numeric(df['water_temp_avg'], errors='coerce')
    df['wave_height_m']  = pd.to_numeric(df['wave_height_m'], errors='coerce')
    df['size_min_cm']    = pd.to_numeric(df['size_min_cm'], errors='coerce')
    df['size_max_cm']    = pd.to_numeric(df['size_max_cm'], errors='coerce')
    df['month']          = df['date'].dt.month
    df['season']         = df['month'].map(_month_to_season)

    df['species_detail'] = df['species']
    return df


def _explode_fish_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """サイズ範囲を匹数で等分して1匹ごとのレコードに展開する。

    例: 30〜40cm 3匹 → size_cm = [30.0, 35.0, 40.0] の3行
    グレは size_cm < 30 → コッパグレ / >= 30 → グレ で再分類。
    """
    if df.empty:
        return df.assign(size_cm=pd.Series(dtype=float))

    counts = df['count'].fillna(1).clip(lower=1).astype(int)
    df_exp = df.loc[df.index.repeat(counts)].copy().reset_index(drop=True)

    ranks = np.concatenate([np.arange(int(c)) for c in counts])
    df_exp['_rank'] = ranks
    df_exp['_cnt']  = counts.values.repeat(counts.values)

    s_min = df_exp['size_min_cm'].values.astype(float)
    s_max = df_exp['size_max_cm'].values.astype(float)
    cnt   = df_exp['_cnt'].values.astype(float)
    rank  = df_exp['_rank'].values.astype(float)

    size_cm   = np.full(len(df_exp), np.nan)
    has_both  = ~np.isnan(s_min) & ~np.isnan(s_max)
    multi     = has_both & (cnt > 1)
    single    = has_both & (cnt == 1)
    only_min  = ~np.isnan(s_min) & np.isnan(s_max)
    only_max  = np.isnan(s_min)  & ~np.isnan(s_max)

    safe_cnt = np.where(cnt > 1, cnt, 2)
    size_cm[multi]    = s_min[multi] + (s_max[multi] - s_min[multi]) * rank[multi] / (safe_cnt[multi] - 1)
    size_cm[single]   = (s_min[single] + s_max[single]) / 2
    size_cm[only_min] = s_min[only_min]
    size_cm[only_max] = s_max[only_max]

    df_exp['size_cm'] = size_cm

    # グレ/コッパグレ を per-fish サイズで再分類
    is_gure = df_exp['species'] == 'グレ'
    df_exp.loc[is_gure & (df_exp['size_cm'] <  30), 'species_detail'] = 'コッパグレ'
    df_exp.loc[is_gure & (df_exp['size_cm'] >= 30), 'species_detail'] = 'グレ'

    return df_exp.drop(columns=['_rank', '_cnt'])


def _split_gure_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """グレレコードをサイズ比率でコッパグレ/グレに分割する。

    - size_max < 30            : コッパグレ（全匹）
    - size_min >= 30           : グレ（全匹）
    - size_min < 30 <= size_max: 比率で按分、どちらかが0匹なら片方のみ
    - サイズなし               : グレのまま
    """
    mask_gure = df['species'] == 'グレ'
    df_other = df[~mask_gure].copy()
    df_gure  = df[mask_gure].copy()

    if df_gure.empty:
        df['species_detail'] = df['species']
        return df

    rows: list[dict] = []
    for _, rec in df_gure.iterrows():
        cnt   = float(rec['count']) if pd.notna(rec.get('count')) and rec['count'] > 0 else 1.0
        s_min = rec.get('size_min_cm')
        s_max = rec.get('size_max_cm')

        if pd.notna(s_min) and pd.notna(s_max) and s_min < 30 <= s_max:
            ratio_koppa = (30.0 - s_min) / (s_max - s_min)
            cnt_koppa   = round(cnt * ratio_koppa)
            cnt_gure    = round(cnt) - cnt_koppa
            if cnt_koppa > 0:
                r = rec.to_dict()
                r['species_detail'] = 'コッパグレ'
                r['count']          = cnt_koppa
                rows.append(r)
            if cnt_gure > 0:
                r = rec.to_dict()
                r['species_detail'] = 'グレ'
                r['count']          = cnt_gure
                rows.append(r)
        elif pd.notna(s_max) and s_max < 30:
            r = rec.to_dict(); r['species_detail'] = 'コッパグレ'; rows.append(r)
        else:
            r = rec.to_dict(); r['species_detail'] = 'グレ'; rows.append(r)

    df_gure_split = pd.DataFrame(rows) if rows else pd.DataFrame(columns=df.columns)
    return pd.concat([df_other, df_gure_split], ignore_index=True)


@st.cache_data(ttl=300)
def load_data_tab2() -> pd.DataFrame:
    """Tab2専用: グレをサイズ比でコッパグレ/グレに分割したDataFrame。"""
    df = load_data()
    if df.empty:
        return df
    return _split_gure_by_size(df)


@st.cache_data(ttl=300)
def load_data_exploded() -> pd.DataFrame:
    """サイズ分析用：1匹1行に展開したDataFrame。"""
    df = load_data()
    if df.empty:
        return df
    src = df[df['count'].notna() & (df['count'] > 0)].copy()
    return _explode_fish_sizes(src)


def _month_to_season(m: int) -> str:
    if m in (3, 4, 5):    return '春'
    if m in (6, 7, 8):    return '夏'
    if m in (9, 10, 11):  return '秋'
    return '冬'


# ---------------------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div style="padding: 8px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom: 20px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-size:1.6rem;">🎣</span>
    <div>
      <div style="color:#FFFFFF; font-size:1.1rem; font-weight:700; line-height:1.2;">谷口渡船</div>
      <div style="color:rgba(255,255,255,0.55); font-size:0.75rem; margin-top:2px;">釣果ダッシュボード</div>
    </div>
  </div>
</div>
<div style="color:rgba(255,255,255,0.5); font-size:0.72rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:10px;">
  絞り込み
</div>
""", unsafe_allow_html=True)

df_all = load_data()

if df_all.empty:
    st.warning('釣果データがありません。ローカルで `python scraper.py` を実行してください。')
    st.stop()

# 期間フィルタ
min_date = df_all['date'].min().date()
max_date = df_all['date'].max().date()
date_range = st.sidebar.date_input(
    '期間', value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    d_from, d_to = date_range
else:
    d_from = d_to = date_range[0] if date_range else min_date

# 魚種フィルタ
all_species = sorted(df_all['species'].dropna().unique())
selected_species = st.sidebar.multiselect('魚種', all_species, default=all_species)


# フィルタ適用
df = df_all[
    (df_all['date'].dt.date >= d_from) &
    (df_all['date'].dt.date <= d_to) &
    (df_all['species'].isin(selected_species))
].copy()

# ---------------------------------------------------------------------------
# ヒーローヘッダー & サマリー指標
# ---------------------------------------------------------------------------

st.markdown("""
<div style="
  background: linear-gradient(135deg, #0B3D5C 0%, #1B8FA8 60%, #22AECB 100%);
  border-radius: 20px;
  padding: 28px 36px;
  margin-bottom: 22px;
  box-shadow: 0 8px 32px rgba(11,61,92,0.25);
  position: relative;
  overflow: hidden;
">
  <div style="
    position:absolute; right:-20px; top:-20px;
    width:180px; height:180px; border-radius:50%;
    background:rgba(255,255,255,0.05);
  "></div>
  <div style="
    position:absolute; right:60px; bottom:-40px;
    width:120px; height:120px; border-radius:50%;
    background:rgba(255,255,255,0.06);
  "></div>
  <div style="position:relative; z-index:1;">
    <h1 style="color:#FFFFFF !important; margin:0 0 6px 0; font-size:1.75rem; font-weight:700; letter-spacing:-0.01em;">
      🎣 谷口渡船 釣果ダッシュボード
    </h1>
    <p style="color:rgba(255,255,255,0.72); margin:0; font-size:0.92rem; font-weight:400;">
      磯釣り釣果データの分析・可視化・釣行予測
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# サマリー指標（KPIカード）
_kpi_total   = int(df_all['count'].sum())
_kpi_days    = df_all['date'].dt.date.nunique()
_kpi_spots   = df_all['spot'].nunique()
_kpi_maxsize = df_all['size_max_cm'].max()

_c1, _c2 = st.columns(2)
_c3, _c4 = st.columns(2)
with _c1:
    st.metric('総釣果数', f'{_kpi_total:,} 匹')
with _c2:
    st.metric('釣行日数', f'{_kpi_days:,} 日')
with _c3:
    st.metric('釣り場数', f'{_kpi_spots} 磯')
with _c4:
    st.metric('最大サイズ', f'{_kpi_maxsize:.0f} cm' if pd.notna(_kpi_maxsize) else '―')

st.markdown('<div style="margin-bottom:8px;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '🏠 今日の判断',
    '🎣 釣果予測',
    '📊 釣果分析',
    '🏆 釣り場ランキング',
    '📅 月別・季節別',
    '🌊 波高・天候条件',
])


# ============================================================
# キャッシュ付きデータローダー（Tab 1 で使用）
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _load_weather(location: str) -> pd.DataFrame | None:
    return tenki_scraper.get_weather_forecast(location)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_morning_wind(location: str) -> pd.DataFrame | None:
    return tenki_scraper.get_morning_wind(location)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_forecast_wind(location: str) -> pd.DataFrame | None:
    return tenki_scraper.get_forecast_wind_risk(location)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_tide(location: str) -> pd.DataFrame | None:
    return tenki_scraper.get_tide(location, days=14)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_hourly() -> pd.DataFrame | None:
    return weather_api.get_hourly_forecast(days=14)

_PREDICTIONS_PATH = Path('ml/predictions.json')

@st.cache_data(ttl=1800, show_spinner=False)
def _load_ai_predictions(days: int = 7) -> list[dict]:
    """予測をml/predictions.jsonから読み込む。なければライブ推論にフォールバック。"""
    if _PREDICTIONS_PATH.exists():
        import json as _json
        payload = _json.loads(_PREDICTIONS_PATH.read_text(encoding='utf-8'))
        preds = payload.get('predictions', [])
        for p in preds:
            if isinstance(p.get('date'), str):
                from datetime import date as _date
                p['date'] = _date.fromisoformat(p['date'])
        return preds
    # ローカル環境でモデルがある場合のみフォールバック
    if not _ML_AVAILABLE or not ml_predict.models_exist():
        return []
    return ml_predict.predict_multi_days(days=days)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_species_base_rates() -> dict[str, float]:
    """魚種別ベースレート（全釣行日のうち1匹以上釣れた日の割合）を返す。"""
    df = load_data()
    if df.empty:
        return {}
    total_days = df['date'].dt.date.nunique()
    if total_days == 0:
        return {}
    rates = (
        df[df['count'] > 0]
        .groupby('species')['date']
        .apply(lambda s: s.dt.date.nunique())
        / total_days
    ).to_dict()
    return rates


@st.cache_data(ttl=3600, show_spinner=False)
def _load_trending_species(
    recent_days: int = 14,
    prev_days: int = 28,
    min_ratio: float = 1.3,
    min_recent_rate: float = 0.05,
) -> pd.DataFrame:
    """直近 recent_days と その前 prev_days の出現率を比較し、急増している魚種を返す。

    Returns:
        columns: species, recent_rate, prev_rate, ratio, label
        ratio 降順ソート済み。min_ratio 未満は除外。
    """
    df = load_data()
    if df.empty:
        return pd.DataFrame()

    latest = df['date'].dt.date.max()
    recent_end   = latest
    recent_start = latest - timedelta(days=recent_days - 1)
    prev_end     = recent_start - timedelta(days=1)
    prev_start   = prev_end - timedelta(days=prev_days - 1)

    def _period_rates(start, end):
        mask = (df['date'].dt.date >= start) & (df['date'].dt.date <= end)
        total = df[mask]['date'].dt.date.nunique()
        if total == 0:
            return pd.Series(dtype=float), 0
        hits = (
            df[mask & (df['count'] > 0)]
            .groupby('species')['date']
            .apply(lambda s: s.dt.date.nunique())
        )
        return hits / total, total

    recent_rates, r_days = _period_rates(recent_start, recent_end)
    prev_rates,   p_days = _period_rates(prev_start,   prev_end)

    if r_days == 0:
        return pd.DataFrame()

    all_sp = set(recent_rates.index) | set(prev_rates.index)
    rows = []
    for sp in all_sp:
        r = float(recent_rates.get(sp, 0))
        p = float(prev_rates.get(sp, 0))
        if r < min_recent_rate:
            continue
        if p < 0.01:
            ratio = 5.0 if r >= min_recent_rate else 0.0  # 前期間ゼロからの初登場
        else:
            ratio = r / p
        if ratio < min_ratio:
            continue
        if ratio >= 3.0:
            label = '🔥 急増中'
        elif ratio >= 2.0:
            label = '📈 増加中'
        else:
            label = '↑ 上昇傾向'
        rows.append({
            'species':     sp,
            'recent_rate': round(r, 3),
            'prev_rate':   round(p, 3),
            'ratio':       round(ratio, 2),
            'label':       label,
            'recent_days': r_days,
            'recent_start': recent_start,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('ratio', ascending=False).reset_index(drop=True)


def _species_lift_str(species_proba: dict[str, float], base_rates: dict[str, float],
                      top_n: int = 3) -> str:
    """species_proba をベースレートで割ったリフト値でランク付けして表示文字列を返す。

    リフト = 今日の予測確率 / 歴史的ベースレート
    ベースレート不明の魚種はリフト1.0として扱う。
    """
    if not species_proba:
        return 'データ不足'
    lifts = {}
    for sp, p in species_proba.items():
        br = base_rates.get(sp, None)
        if br and br > 0:
            lifts[sp] = p / br
        else:
            lifts[sp] = 1.0  # ベースレート不明はニュートラル
    top = sorted(lifts.items(), key=lambda x: -x[1])[:top_n]
    return '・'.join(f"{sp}({v:.1f}x)" for sp, v in top)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_forecast_api() -> pd.DataFrame | None:
    return weather_api.get_forecast()

@st.cache_data(ttl=1800, show_spinner=False)
def _load_spot_ranking(target_date_str: str) -> pd.DataFrame:
    if not _ML_AVAILABLE or not ml_predict.models_exist():
        return pd.DataFrame()
    d = date.fromisoformat(target_date_str)
    return ml_predict.predict_spot_ranking(d)


def _fmt_tide_list(rows: pd.DataFrame) -> str:
    return '　'.join(f"{r['time']}({r['height_cm']}cm)" for _, r in rows.iterrows())


# ============================================================
# Tab 0: 今日の判断（ホーム）
# ============================================================
with tab0:
    _today = date.today()
    _WDAYS = '月火水木金土日'

    # ── データ取得（キャッシュ利用）────────────────────────────
    _h_ai   = _load_ai_predictions(days=7)

    # 予測生成日時を表示
    if _PREDICTIONS_PATH.exists():
        import json as _json
        _gen_at = _json.loads(_PREDICTIONS_PATH.read_text(encoding='utf-8')).get('generated_at', '')
        if _gen_at:
            st.caption(f'AI予測更新日時: {_gen_at}')
    _h_br   = _load_species_base_rates()
    _h_wx   = _load_weather('串本')
    _h_tide = _load_tide('串本')
    _h_fc   = _load_forecast_api()
    _h_mw   = _load_morning_wind('串本')   # tenki.jp 0〜6時風速（3日分）
    _h_fw   = _load_forecast_wind('串本')  # tenki.jp 10days 日別風速（4日目〜）

    # ── 日付選択（クリックで詳細表示）──────────────────────────────
    _ai_dates = [p['date'] for p in _h_ai] if _h_ai else [_today]
    _default_date = _today if _today in _ai_dates else _ai_dates[0] if _ai_dates else _today

    if 'home_sel_date' not in st.session_state or st.session_state['home_sel_date'] not in _ai_dates:
        st.session_state['home_sel_date'] = _default_date

    if _h_ai:
        # 風速リスクマップ: 10days（全日）を先に入れ、0〜6時データ（精度高）で上書き
        _mw_map = {}
        if _h_fw is not None and not _h_fw.empty:
            for _, _mwr in _h_fw.iterrows():
                _mw_map[_mwr['date']] = _mwr
        if _h_mw is not None and not _h_mw.empty:
            for _, _mwr in _h_mw.iterrows():
                _mw_map[_mwr['date']] = _mwr  # 上書き（より精度高い）

        # テーブルヘッダー（3列: 判定 | 日付+休船リスク | スコア+期待釣果）
        _hcols = st.columns([2, 5, 3])
        for _hc, _ht in zip(_hcols, ['判定', '日付 / 休船リスク', 'スコア / 釣果']):
            _hc.markdown(f'<span style="font-size:0.8rem; color:#888; font-weight:600;">{_ht}</span>',
                         unsafe_allow_html=True)
        st.markdown('<hr style="margin:4px 0 8px;">', unsafe_allow_html=True)

        for _p in _h_ai:
            _d  = _p['date']
            _wd = _WDAYS[_d.weekday()]
            _gp = _p.get('go_proba', 0)
            _is_today = (_d == _today)
            _is_sel   = (st.session_state['home_sel_date'] == _d)

            # 休船リスク（tenki.jp 0〜6時風速）→ 判定基準
            _mw_row = _mw_map.get(_d)
            if _mw_row is not None:
                _mw_spd  = _mw_row['wind_max_ms']
                _mw_prob = _mw_row['risk_prob']
                if _mw_prob >= 0.90:
                    _verdict_sym, _v_color, _v_bg = '✖ STOP',   '#7a1c24', '#f8d7da'
                    _mw_str,  _mw_color = f'⚠ 休船確率大({int(_mw_spd)}m/s)', '#7a1c24'
                elif _mw_prob >= 0.75:
                    _verdict_sym, _v_color, _v_bg = '⚠️ CHECK', '#7a5f00', '#fff3cd'
                    _mw_str,  _mw_color = f'△ 可能性あり({int(_mw_spd)}m/s)', '#7a5f00'
                else:
                    _verdict_sym, _v_color, _v_bg = '✅ GO',    '#1a7a4a', '#d4edda'
                    _mw_str,  _mw_color = f'○ 出船可({int(_mw_spd)}m/s)', '#1a7a4a'
            else:
                _verdict_sym, _v_color, _v_bg = '✅ GO',    '#1a7a4a', '#d4edda'
                _mw_str, _mw_color = '–', '#aaa'

            _bold     = 'font-weight:700;' if _is_sel else ''
            _date_str = f"{_d.month}/{_d.day}({_wd})" + (' ★今日' if _is_today else '')
            _exp_str  = f"{_p.get('expected_count', 0):.1f}匹"

            _rcols = st.columns([2, 5, 3])
            with _rcols[0]:
                if st.button(
                    _verdict_sym,
                    key=f'home_day_{_d}',
                    help=f"{_d.month}/{_d.day} の詳細を表示",
                    use_container_width=True,
                ):
                    st.session_state['home_sel_date'] = _d
                    st.rerun()
            _rcols[1].markdown(
                f'<span style="font-size:0.92rem;{_bold}">{_date_str}</span><br>'
                f'<span style="font-size:0.78rem; color:{_mw_color};">{_mw_str}</span>',
                unsafe_allow_html=True,
            )
            _rcols[2].markdown(
                f'<span style="font-size:0.92rem; color:{_v_color};{_bold}">{_gp*100:.0f}%</span><br>'
                f'<span style="font-size:0.78rem; color:#555;">{_exp_str}</span>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="margin:8px 0 16px;">', unsafe_allow_html=True)
        _sel_date = st.session_state['home_sel_date']

        # 選択中の日付を見出しで表示
        _sd_p = next((p for p in _h_ai if p['date'] == _sel_date), None)
        if _sd_p:
            _sd_mw = _mw_map.get(_sel_date)
            _sd_rp = _sd_mw['risk_prob'] if _sd_mw is not None else 0.0
            if _sd_rp >= 0.90:
                _sd_sym, _sd_col = '✖ STOP',   '#7a1c24'
            elif _sd_rp >= 0.75:
                _sd_sym, _sd_col = '⚠️ CHECK', '#7a5f00'
            else:
                _sd_sym, _sd_col = '✅ GO',    '#1a7a4a'
            st.markdown(
                f'<div style="font-size:1.05rem; font-weight:700; color:{_sd_col}; margin-bottom:8px;">'
                f'▶ {_sel_date.month}/{_sel_date.day}({_WDAYS[_sel_date.weekday()]}) — {_sd_sym}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info('AI予測データがありません。')
        _sel_date = _today

    _sel_pred = next((p for p in _h_ai if p['date'] == _sel_date), None) if _h_ai else None
    _go_proba  = _sel_pred.get('go_proba') if _sel_pred else None
    _exp_count = _sel_pred.get('expected_count') if _sel_pred else None

    st.markdown('---')

    # ── GO / CHECK / STOP バナー ────────────────────────────────
    _sel_mw  = _mw_map.get(_sel_date)
    _sel_rp  = _sel_mw['risk_prob'] if _sel_mw is not None else 0.0
    if _go_proba is not None or _sel_mw is not None:
        if _sel_rp >= 0.90:
            _verdict, _vc, _vbg, _vmsg = '✖ STOP',   '#7a1c24', '#f8d7da', '出船困難な可能性があります'
        elif _sel_rp >= 0.75:
            _verdict, _vc, _vbg, _vmsg = '⚠️ CHECK', '#7a5f00', '#fff3cd', '出船できない可能性があります。確認してください'
        else:
            _verdict, _vc, _vbg, _vmsg = '✅ GO',    '#1a7a4a', '#d4edda', '出船できる見込みです'

        _sp_sel = _species_lift_str(_sel_pred.get('species_proba', {}), _h_br, top_n=2)
        st.markdown(f"""
<div style="background:{_vbg}; border-left:6px solid {_vc};
     padding:20px 24px; border-radius:10px; margin-bottom:20px;">
  <div style="font-size:2rem; font-weight:800; color:{_vc}; line-height:1.1;">{_verdict}</div>
  <div style="font-size:0.95rem; color:{_vc}; margin-top:4px;">{_vmsg}</div>
  <div style="margin-top:14px; display:flex; flex-wrap:wrap; gap:24px; color:#333;">
    <span>推奨スコア&nbsp;<b>{_go_proba*100:.0f}%</b></span>
    <span>期待釣果&nbsp;<b>{_exp_count:.1f}&nbsp;匹</b></span>
    <span>注目魚種&nbsp;<b>{_sp_sel}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── コンディション（4カード）────────────────────────────────
    _sel_wx_row = None
    if _h_wx is not None:
        _wr = _h_wx[_h_wx['date'] == _sel_date]
        _sel_wx_row = _wr.iloc[0] if not _wr.empty else None

    _sel_fc_row = None
    if _h_fc is not None:
        _fr = _h_fc[_h_fc['date'] == _sel_date]
        _sel_fc_row = _fr.iloc[0] if not _fr.empty else None

    _sel_tide_name = None
    _sel_rising    = _sel_pred.get('rising_ratio') if _sel_pred else None
    if _h_tide is not None and not _h_tide.empty:
        _tdn = _h_tide[(_h_tide['date'] == _sel_date) & _h_tide['tide_name'].notna()]
        if not _tdn.empty:
            _sel_tide_name = str(_tdn.iloc[0]['tide_name'])

    _recent_wt = df_all[
        df_all['date'] >= (df_all['date'].max() - pd.Timedelta(days=7))
    ]['water_temp_avg'].mean()

    # 天気: predictions.json → _h_wx の順で取得
    _weather_raw = (_sel_pred.get('weather') if _sel_pred else None)
    if not _weather_raw and _sel_wx_row is not None and pd.notna(_sel_wx_row.get('weather')):
        _weather_raw = str(_sel_wx_row['weather'])
    _weather_str = _weather_raw or '–'

    # Open-Meteo の windspeed_10m_max は km/h 単位のため m/s に変換（÷ 3.6）
    _wind_ms  = (float(_sel_pred['wind_ms_max']) / 3.6
                 if _sel_pred and _sel_pred.get('wind_ms_max') is not None
                    and not pd.isna(_sel_pred.get('wind_ms_max', float('nan'))) else None)
    _wave_val = (float(_sel_fc_row['forecast_wave_height_m'])
                 if _sel_fc_row is not None and pd.notna(_sel_fc_row.get('forecast_wave_height_m')) else None)
    _wave_str = f"{_wave_val:.1f} m" if _wave_val is not None else '–'
    _wt_str   = f"{_recent_wt:.1f} ℃" if pd.notna(_recent_wt) else '–'
    _tide_str = _sel_tide_name or (_sel_pred.get('tide_name') if _sel_pred else None) or '–'

    _mc1, _mc2 = st.columns(2)
    _mc3, _mc4 = st.columns(2)
    with _mc1: st.metric('☁️ 天気', _weather_str)
    with _mc2: st.metric('🌊 波高（予報）', _wave_str)
    with _mc3: st.metric('🌡️ 水温（直近7日）', _wt_str)
    with _mc4: st.metric('🌙 潮', _tide_str)

    st.markdown('---')

    # ── 見送り・推奨理由 ─────────────────────────────────────────
    def _build_reasons(pred, wind_ms, wave_val, wt, tide_name, rising):
        """ネガティブ・ポジティブ要因を (icon, text, is_bad) のリストで返す。"""
        reasons = []

        # 風速
        if wind_ms is not None:
            if wind_ms >= 15:
                reasons.append(('🚨', f'強風 {wind_ms:.1f}m/s — 渡礁・釣行が危険な風速です', True))
            elif wind_ms >= 10:
                reasons.append(('⚠️', f'やや強風 {wind_ms:.1f}m/s — 仕掛けが流されやすくなります', True))
            else:
                reasons.append(('✅', f'風は穏やか {wind_ms:.1f}m/s — 釣りやすい条件です', False))

        # 風向（predに wind_dir_deg が入っている場合）
        if pred:
            _wd_deg = pred.get('wind_dir_deg')
            if _wd_deg is not None and not pd.isna(_wd_deg):
                _wd_deg = float(_wd_deg)
                # 串本の磯は山を背にしているため北〜北西風は影響小
                # 南〜南東風（海側からの風）は磯に直接当たり危険
                if 135 <= _wd_deg <= 225:
                    reasons.append(('⚠️', f'南〜南西風（{_wd_deg:.0f}°）— 海側からの風で磯に直接当たります', True))
                elif 270 <= _wd_deg <= 360 or _wd_deg <= 30:
                    reasons.append(('✅', f'北〜北西風（{_wd_deg:.0f}°）— 山が背後にあり串本の磯は風の影響を受けにくい方向です', False))

        # 波高
        if wave_val is not None:
            if wave_val >= 2.5:
                reasons.append(('🚨', f'高波 {wave_val:.1f}m — 磯釣り困難な波高です', True))
            elif wave_val >= 1.5:
                reasons.append(('⚠️', f'波やや高め {wave_val:.1f}m — 荒れ気味の磯があります', True))
            else:
                reasons.append(('✅', f'波は穏やか {wave_val:.1f}m — 磯釣りに適した波高です', False))

        # 水温
        if wt is not None and not pd.isna(wt):
            if wt < 14:
                reasons.append(('⚠️', f'水温 {wt:.1f}℃ — 低すぎてグレも口を使いにくくなります', True))
            elif wt >= 15 and wt <= 22:
                reasons.append(('✅', f'水温 {wt:.1f}℃ — グレの適水温帯です', False))

        # 潮
        if tide_name:
            if tide_name in ('大潮',):
                reasons.append(('✅', f'潮は{tide_name} — 潮の動きが最も活発で好条件です', False))
            elif tide_name in ('長潮', '若潮'):
                reasons.append(('⚠️', f'潮は{tide_name} — 潮の動きが弱く釣果が落ちやすい時期です', True))

        # 上り潮割合
        if rising is not None and not pd.isna(rising):
            rising = float(rising)
            if rising >= 0.6:
                reasons.append(('✅', f'上り潮が多い（{rising*100:.0f}%）— 朝〜昼の釣りに有利です', False))
            elif rising <= 0.25:
                reasons.append(('⚠️', f'上り潮が少ない（{rising*100:.0f}%）— 下り潮中心の時間帯です', True))

        # 降水
        if pred:
            p1 = pred.get('precip_1d', 0) or 0
            p2 = pred.get('precip_2d', 0) or 0
            p3 = pred.get('precip_3d', 0) or 0
            if p1 > 5 or p2 > 5 or p3 > 5:
                reasons.append(('⚠️', f'直近に降雨あり（前日{p1:.0f}mm / 前々日{p2:.0f}mm）— 濁りが入っている可能性があります', True))
            elif p1 == 0 and p2 == 0:
                reasons.append(('✅', '直近3日間は無雨 — 海の状態は安定しています', False))

        # 総合スコア
        if pred:
            gp = pred.get('go_proba', 0)
            if gp < 0.3:
                reasons.append(('✖', f'AI総合判定 {gp*100:.0f}% — 過去の同条件と比較してかなり悪い日です', True))
            elif gp >= 0.6:
                reasons.append(('✅', f'AI総合判定 {gp*100:.0f}% — 過去の同条件と比較して良い日です', False))

        return reasons

    _reasons = _build_reasons(
        pred=_sel_pred,
        wind_ms=_wind_ms,
        wave_val=_wave_val,
        wt=_recent_wt if pd.notna(_recent_wt) else None,
        tide_name=_sel_tide_name,
        rising=_sel_rising,
    )

    _bad  = [(i, t) for i, t, b in _reasons if b]
    _good = [(i, t) for i, t, b in _reasons if not b]

    _r1, _r2 = st.columns(2)
    with _r1:
        st.markdown('##### ⚠️ 懸念事項' if _bad else '##### ✅ 懸念事項なし')
        if _bad:
            for _icon, _txt in _bad:
                st.markdown(f'{_icon} {_txt}')
        else:
            st.markdown('現在の予報では大きな懸念点はありません。')
    with _r2:
        st.markdown('##### ✅ 好条件の理由' if _good else '##### ℹ️ 好条件なし')
        if _good:
            for _icon, _txt in _good:
                st.markdown(f'{_icon} {_txt}')
        else:
            st.markdown('この日は好条件の要素が少ない状況です。')

    st.markdown('---')

    # ── 磯ランキングTOP5 ＋ 魚種リフト ──────────────────────────
    _rc, _sc = st.columns([3, 2])

    with _rc:
        st.markdown(f'##### 🏆 {_sel_date.month}/{_sel_date.day} の磯 期待度ランキング TOP5')
        with st.spinner('計算中...'):
            _home_rank = _load_spot_ranking(_sel_date.isoformat())

        if _home_rank.empty:
            st.info('磯ランキングデータがありません。')
        else:
            for _ri, _rrow in _home_rank.head(5).iterrows():
                _rn = _ri + 1
                _medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(_rn, f'{_rn}.')
                st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; padding:10px 0;
     border-bottom:1px solid #eee;">
  <span style="font-size:1.15rem; min-width:30px;">{_medal}</span>
  <span style="font-weight:600; flex:1;">{_rrow['spot']}</span>
  <span style="color:#e07b39; font-weight:700; font-size:1.05rem;">{_rrow['expected_count']:.1f} 匹</span>
  <span style="color:#888; font-size:0.85rem; min-width:40px; text-align:right;">{_rrow['go_proba']*100:.0f}%</span>
</div>""", unsafe_allow_html=True)
        st.caption('詳細は「🏆 釣り場ランキング」タブへ')

    with _sc:
        st.markdown('##### 🐟 注目魚種（平均比）')
        if _sel_pred:
            _sp_raw   = _sel_pred.get('species_proba', {})
            _sp_lifts = {
                sp: p / _h_br[sp] if _h_br.get(sp, 0) > 0 else 1.0
                for sp, p in _sp_raw.items()
            }
            for _sp, _lv in sorted(_sp_lifts.items(), key=lambda x: -x[1]):
                _sc_color = '#1a7a4a' if _lv >= 1.2 else ('#7a5f00' if _lv >= 0.8 else '#999')
                _bar_w = min(int(_lv / 2.0 * 100), 100)
                st.markdown(f"""
<div style="margin-bottom:10px;">
  <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
    <span style="font-weight:600;">{_sp}</span>
    <span style="color:{_sc_color}; font-weight:700;">{_lv:.2f}x</span>
  </div>
  <div style="background:#eee; border-radius:4px; height:6px;">
    <div style="background:{_sc_color}; width:{_bar_w}%; height:6px; border-radius:4px;"></div>
  </div>
</div>""", unsafe_allow_html=True)
            st.caption('1.0x = 平均並み　2.0x = 平均の2倍釣れやすい')
        else:
            st.info('AIモデルが必要です。')

    st.markdown('---')

    # ── 最近釣れ始めている魚 ─────────────────────────────────────
    st.markdown('##### 🐟 最近釣れ始めている魚')

    _trending = _load_trending_species()

    if _trending.empty:
        st.info('直近のデータから急増している魚種は検出されませんでした。')
    else:
        _ref_start = _trending.iloc[0]['recent_start']
        _ref_days  = _trending.iloc[0]['recent_days']
        st.caption(
            f'直近{_ref_days}日間（{_ref_start.month}/{_ref_start.day}〜）の出現率が'
            'それ以前の28日間と比べて1.3倍以上になっている魚種です。'
        )

        _tr_cols = st.columns(min(len(_trending), 4))
        for _ci, (_, _row) in enumerate(zip(_tr_cols, _trending.itertuples())):
            with _tr_cols[_ci]:
                _r_pct = _row.recent_rate * 100
                _p_pct = _row.prev_rate  * 100
                _ratio = _row.ratio
                _lbl   = _row.label
                _lbl_color = '#c0392b' if '急増' in _lbl else ('#2471a3' if '増加' in _lbl else '#1a7a4a')
                st.markdown(f"""
<div style="background:#FFFFFF; border-radius:12px; padding:16px 18px;
     box-shadow:0 2px 12px rgba(0,0,0,0.07); border-top:4px solid {_lbl_color};">
  <div style="font-size:0.78rem; font-weight:600; color:{_lbl_color};
       letter-spacing:0.05em; margin-bottom:6px;">{_lbl}</div>
  <div style="font-size:1.25rem; font-weight:700; color:#0B3D5C;
       margin-bottom:10px;">{_row.species}</div>
  <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#555;">
    <span>直近</span><span style="font-weight:700; color:{_lbl_color};">{_r_pct:.0f}%</span>
  </div>
  <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#888;">
    <span>前期間</span><span>{_p_pct:.0f}%</span>
  </div>
  <div style="margin-top:8px; background:#eee; border-radius:4px; height:5px; overflow:hidden;">
    <div style="background:{_lbl_color}; width:{min(_r_pct*2, 100):.0f}%;
         height:5px; border-radius:4px;"></div>
  </div>
  <div style="margin-top:6px; font-size:0.75rem; color:#999; text-align:right;">
    前期間比 {_ratio:.1f}x
  </div>
</div>""", unsafe_allow_html=True)

    st.caption('詳細な釣行計画は「🎣 釣果予測」タブ、過去データ分析は「📊 釣果分析」タブをご利用ください。')


# ============================================================
# Tab 1: 釣果予測
# ============================================================
with tab1:
    st.subheader('釣果予測')
    location = st.radio('地点', ['串本', '白浜'], horizontal=True)

    with st.spinner('データ取得中...'):
        wx_df     = _load_weather(location)
        tide_df   = _load_tide(location)
        hourly_df = _load_hourly()

    # ── AI 予測セクション ─────────────────────────────────────
    st.markdown('##### 🤖 AI 釣行予測（向こう7日間）')

    ai_predictions = _load_ai_predictions(days=7)

    if not ai_predictions:
        st.warning('AI予測データを取得できませんでした。')
    else:
        # 7日間テーブル
        ai_rows = []
        _WEEKDAYS = '月火水木金土日'
        _base_rates = _load_species_base_rates()
        for pred in ai_predictions:
            d = pred['date']
            wd = _WEEKDAYS[d.weekday()]
            sp_str = _species_lift_str(pred.get('species_proba', {}), _base_rates)
            ai_rows.append({
                '日付':               f'{d.month}/{d.day}({wd})',
                '推奨スコア':          f"{pred.get('go_proba', 0)*100:.0f}%",
                '期待釣果数（匹）':    pred.get('expected_count', '–'),
                '釣れやすい魚種TOP3 (平均比)': sp_str,
            })
        st.dataframe(pd.DataFrame(ai_rows), use_container_width=True, hide_index=True)

        # 日付選択 → 磯別ランキング + Claudeプロンプト
        with st.expander('🏆 磯ランキング・Claude アドバイスを見る', expanded=False):
          ai_dates = [p['date'] for p in ai_predictions]
          selected_pred_date = st.selectbox(
              '日付を選択',
              options=ai_dates,
              format_func=lambda d: f"{d.month}/{d.day}({_WEEKDAYS[d.weekday()]})",
          )
          selected_pred = next((p for p in ai_predictions if p['date'] == selected_pred_date), None)

          st.markdown('###### 🏆 磯の期待度ランキング（上位10磯）')
          with st.spinner('磯ランキング計算中...'):
              rank_df = _load_spot_ranking(selected_pred_date.isoformat())
          if rank_df.empty:
              st.info('磯ランキングデータがありません。')
          else:
              rank_df_disp = rank_df.head(10).copy()
              rank_df_disp['expected_count'] = rank_df_disp['expected_count'].round(1)
              rank_df_disp['go_proba']       = (rank_df_disp['go_proba'] * 100).round(1)
              rank_df_disp = rank_df_disp.rename(columns={
                  'spot':           '釣り場',
                  'expected_count': '期待釣果（匹）',
                  'go_proba':       '推奨スコア（%）',
              })
              st.dataframe(rank_df_disp, use_container_width=True, hide_index=True)

          st.markdown('###### 💬 Claude AIへのプロンプト')
          if selected_pred is not None:
              forecast_df_api = _load_forecast_api()
              fc_row = None
              if forecast_df_api is not None:
                  fc_match = forecast_df_api[
                      forecast_df_api['date'] == selected_pred_date
                  ]
                  fc_row = fc_match.iloc[0] if not fc_match.empty else None

              tide_day = None
              if tide_df is not None and not tide_df.empty:
                  td = tide_df[tide_df['date'] == selected_pred_date]
                  if not td.empty:
                      tide_day = td.iloc[0].get('tide_name', '')

              # 天気: predictions.json → wx_df の順で取得
              _pred_weather  = selected_pred.get('weather')
              _pred_temp_max = selected_pred.get('temp_max')
              _pred_temp_min = selected_pred.get('temp_min')
              if not _pred_weather and wx_df is not None:
                  wd_rows = wx_df[wx_df['date'] == selected_pred_date]
                  if not wd_rows.empty:
                      wx_day = wd_rows.iloc[0]
                      _pred_weather  = _pred_weather  or (str(wx_day['weather'])  if pd.notna(wx_day.get('weather'))  else None)
                      _pred_temp_max = _pred_temp_max or (float(wx_day['temp_max']) if pd.notna(wx_day.get('temp_max')) else None)
                      _pred_temp_min = _pred_temp_min or (float(wx_day['temp_min']) if pd.notna(wx_day.get('temp_min')) else None)

              recent_wt = (
                  df_all[df_all['date'] >= (df_all['date'].max() - pd.Timedelta(days=7))]
                  ['water_temp_avg'].mean()
              )
              wt = float(recent_wt) if pd.notna(recent_wt) else None

              _sp_proba_raw = selected_pred.get('species_proba', {})
              _br = _load_species_base_rates()
              species_rank = sorted(
                  [
                      (sp, p / _br[sp] if _br.get(sp, 0) > 0 else 1.0)
                      for sp, p in _sp_proba_raw.items()
                  ],
                  key=lambda x: -x[1],
              )[:3]
              wave_fc = fc_row.get('forecast_wave_height_m') if fc_row is not None else None
              sim_mask = df_all['count'].notna()
              if wave_fc is not None and pd.notna(wave_fc):
                  sim_mask &= df_all['wave_height_m'].between(
                      float(wave_fc) - 0.5, float(wave_fc) + 0.5
                  )
              if wt is not None:
                  sim_mask &= df_all['water_temp_avg'].between(wt - 1, wt + 1)
              sim_df = df_all[sim_mask]

              prompt_text = prompt_builder.build_prompt(
                  target_date    = selected_pred_date,
                  weather        = _pred_weather or '不明',
                  temp_max       = float(_pred_temp_max) if _pred_temp_max is not None else None,
                  temp_min       = float(_pred_temp_min) if _pred_temp_min is not None else None,
                  wind_dir       = selected_pred.get('wind_dir', '不明') if hasattr(selected_pred, 'get') else '不明',
                  wind_speed_ms  = None,
                  wave_height_m  = (float(wave_fc) if wave_fc is not None and pd.notna(wave_fc) else None),
                  water_temp_c   = wt,
                  tide_name      = tide_day or selected_pred.get('tide_name') or '不明',
                  rising_ratio   = selected_pred.get('rising_ratio'),
                  precip_1d      = selected_pred.get('precip_1d'),
                  precip_2d      = selected_pred.get('precip_2d'),
                  precip_3d      = selected_pred.get('precip_3d'),
                  predicted_count= selected_pred.get('expected_count'),
                  go_score_pct   = (selected_pred.get('go_proba', 0) * 100
                                    if selected_pred.get('go_proba') is not None else None),
                  species_rank   = species_rank,
                  n_similar      = len(sim_df),
                  avg_count_similar = (float(sim_df['count'].mean())
                                       if len(sim_df) > 0 else None),
              )
              st.text_area('プロンプト（コピーして claude.ai に貼り付け）',
                           value=prompt_text, height=300,
                           key='ai_prompt_text')
              st.caption('上のテキストをコピーして https://claude.ai に貼り付けてください。')

    st.markdown('---')

    # ── ルールベース 参考テーブル ─────────────────────────────
    with st.expander('📊 ルールベース釣果スコア（参考）', expanded=False):
        forecast_df_rb = _load_forecast_api()
        if forecast_df_rb is None:
            st.warning('Open-Meteo API に接続できませんでした。')
        else:
            recent_temp = df_all[
                df_all['date'] >= (df_all['date'].max() - pd.Timedelta(days=30))
            ]['water_temp_avg'].mean()

            rb_rows = []
            for _, fc in forecast_df_rb.iterrows():
                wave     = fc.get('forecast_wave_height_m')
                temp_ref = recent_temp if pd.notna(recent_temp) else None

                mask = pd.Series([True] * len(df_all), index=df_all.index)
                if pd.notna(wave):
                    mask &= df_all['wave_height_m'].between(wave - 0.5, wave + 0.5)
                if temp_ref is not None:
                    mask &= df_all['water_temp_avg'].between(temp_ref - 1, temp_ref + 1)

                matched = df_all[mask & df_all['count'].notna()]
                score = (f"{matched['count'].mean():.1f} 匹 (n={len(matched)})"
                         if len(matched) >= 3 else 'データ不足')

                rb_rows.append({
                    '日付':          str(fc['date']),
                    '天気コード':     fc.get('forecast_weather_code'),
                    '最大波高(m)':    wave,
                    '最高気温(℃)':   fc.get('forecast_temp_max'),
                    '期待釣果スコア': score,
                })

            st.dataframe(pd.DataFrame(rb_rows), use_container_width=True)

    st.markdown('---')

    # ── 日別天気・潮汐詳細エキスパンダー ──────────────────────
    st.markdown('##### 🗓️ 天気・潮汐詳細（14日間）')
    if location == '白浜':
        st.caption('※ 白浜の潮汐は最寄港「田辺」のデータを表示しています。')

    if wx_df is None and hourly_df is None:
        st.warning('天気データを取得できませんでした。')
    else:
        dates = sorted(set(
            (list(wx_df['date'].unique()) if wx_df is not None else []) +
            (list(hourly_df['date'].unique()) if hourly_df is not None else [])
        ))

        _WEEKDAYS = '月火水木金土日'
        _HOURS = [0, 6, 12, 18]
        _XRANGE = [-3, 21]
        _XTICKS = dict(tickvals=[0, 6, 12, 18],
                       ticktext=['0:00', '6:00', '12:00', '18:00'])
        _CHART_MARGIN = dict(l=45, r=10, t=8, b=8)

        def _build_day_info(d):
            """1日分のテキスト情報をまとめて返す（チャートは含まない）。"""
            if wx_df is not None:
                day_rows_ = wx_df[wx_df['date'] == d]
                day_wx_   = day_rows_.iloc[0] if not day_rows_.empty else None
            else:
                day_rows_ = pd.DataFrame()
                day_wx_   = None

            weather_label_ = (str(day_wx_['weather'])
                               if day_wx_ is not None and pd.notna(day_wx_.get('weather')) else '')
            temp_max_ = day_wx_.get('temp_max') if day_wx_ is not None else None
            temp_min_ = day_wx_.get('temp_min') if day_wx_ is not None else None
            sunrise_  = (str(day_wx_['sunrise'])
                         if day_wx_ is not None and pd.notna(day_wx_.get('sunrise')) else '')

            tide_name_str_ = ''
            high_str_ = low_str_ = ''
            if tide_df is not None and not tide_df.empty:
                day_tide_   = tide_df[tide_df['date'] == d]
                high_tides_ = day_tide_[day_tide_['type'] == '満潮'].sort_values('time')
                low_tides_  = day_tide_[day_tide_['type'] == '干潮'].sort_values('time')
                high_str_   = _fmt_tide_list(high_tides_)
                low_str_    = _fmt_tide_list(low_tides_)
                tn_ = day_tide_['tide_name'].dropna()
                tide_name_str_ = str(tn_.iloc[0]) if not tn_.empty else ''

            return day_rows_, day_wx_, weather_label_, temp_max_, temp_min_, sunrise_, \
                   tide_name_str_, high_str_, low_str_

        # ── 詳細グラフ（選択した1日分のみ描画）────────────────
        detail_col, _ = st.columns([1, 2])
        with detail_col:
            selected_detail_date = st.selectbox(
                '📈 詳細グラフを表示する日',
                options=dates,
                format_func=lambda d: f"{d.month}/{d.day}({_WEEKDAYS[d.weekday()]})",
                key='detail_date_select',
            )

        det_rows, det_wx, det_weather, det_tmax, det_tmin, det_sunrise, \
            det_tidename, det_high, det_low = _build_day_info(selected_detail_date)

        det_hourly = (hourly_df[hourly_df['date'] == selected_detail_date]
                      if hourly_df is not None else pd.DataFrame())

        ch_left, ch_right = st.columns(2)

        with ch_left:
            # 気温グラフ
            temp_pts = []
            for hour in _HOURS:
                slot = (det_hourly[det_hourly['hour'] == hour]
                        if not det_hourly.empty else pd.DataFrame())
                v = slot.iloc[0]['temp_c'] if not slot.empty and pd.notna(slot.iloc[0].get('temp_c')) else None
                temp_pts.append((hour, v))
            if any(v is not None for _, v in temp_pts):
                xs = [h for h, v in temp_pts if v is not None]
                ys = [v for _, v in temp_pts if v is not None]
                fig_t = go.Figure(go.Scatter(
                    x=xs, y=ys, mode='lines+markers',
                    line=dict(shape='spline', color='tomato', width=2),
                    marker=dict(size=7, color='tomato'), showlegend=False,
                ))
                fig_t.update_xaxes(range=_XRANGE, showticklabels=True,
                                   showgrid=True, gridcolor='#eee', zeroline=False, **_XTICKS)
                fig_t.update_yaxes(title_text='℃', tickformat='.0f',
                                   showgrid=True, gridcolor='#eee')
                fig_t.update_layout(height=180, margin=_CHART_MARGIN,
                                    plot_bgcolor='white', paper_bgcolor='white',
                                    title=dict(text='気温', font_size=13, x=0.02))
                st.plotly_chart(fig_t, use_container_width=True)

        with ch_right:
            # 潮汐グラフ
            if tide_df is not None and not tide_df.empty:
                day_tide_plot = tide_df[tide_df['date'] == selected_detail_date].copy()
                if not day_tide_plot.empty:
                    day_tide_plot['height_cm'] = pd.to_numeric(
                        day_tide_plot['height_cm'], errors='coerce')
                    day_tide_plot['hour_f'] = day_tide_plot['time'].apply(
                        lambda t: int(str(t).split(':')[0]) + int(str(t).split(':')[1]) / 60
                    )
                    day_tide_plot = day_tide_plot.sort_values('hour_f')
                    fig_td = go.Figure(go.Scatter(
                        x=day_tide_plot['hour_f'], y=day_tide_plot['height_cm'],
                        mode='lines+markers',
                        line=dict(shape='spline', color='steelblue', width=2),
                        marker=dict(size=7, color='steelblue'),
                        text=day_tide_plot.apply(
                            lambda r: f"{r['type']} {str(r['time'])[:5]} {r['height_cm']:.0f}cm",
                            axis=1,
                        ),
                        hovertemplate='%{text}<extra></extra>', showlegend=False,
                    ))
                    fig_td.update_xaxes(range=_XRANGE, showgrid=True,
                                        gridcolor='#eee', **_XTICKS)
                    fig_td.update_yaxes(title_text='cm', tickformat='.0f',
                                        showgrid=True, gridcolor='#eee')
                    fig_td.update_layout(height=180, margin=_CHART_MARGIN,
                                         plot_bgcolor='white', paper_bgcolor='white',
                                         title=dict(text='潮位', font_size=13, x=0.02))
                    st.plotly_chart(fig_td, use_container_width=True)

        st.markdown('---')

        # ── 各日サマリー（テキストのみ・チャートなし）──────────
        for d in dates:
            day_rows, day_wx, weather_label, temp_max, temp_min, sunrise, \
                tide_name_str, high_str, low_str = _build_day_info(d)

            wd    = _WEEKDAYS[d.weekday()]
            title = f"{d.month}/{d.day}({wd})"
            if weather_label:
                title += f"　{weather_label}"
            if temp_max is not None and temp_min is not None:
                title += f"　🌡️{int(temp_max)}/{int(temp_min)}℃"
            elif temp_max is not None:
                title += f"　🌡️{int(temp_max)}℃"
            if tide_name_str:
                title += f"　🌙{tide_name_str}"
            if high_str:
                title += f"　🌊満潮 {high_str}"
            if low_str:
                title += f"　干潮 {low_str}"
            if sunrise:
                title += f"　🌅{sunrise}"

            with st.expander(title):
                day_hourly = (hourly_df[hourly_df['date'] == d]
                              if hourly_df is not None else pd.DataFrame())

                # ── 時刻別明細（2列×2行）──
                for row_start in (0, 2):
                    cols = st.columns(2)
                    for col_idx, hour in enumerate(_HOURS[row_start:row_start + 2]):
                        slot = (day_hourly[day_hourly['hour'] == hour]
                                if not day_hourly.empty else pd.DataFrame())
                        hr_wx = (day_rows[day_rows['hour'] == hour]
                                 if day_wx is not None and not day_rows.empty
                                 else pd.DataFrame())
                        with cols[col_idx]:
                            st.markdown(f"**{hour:02d}:00**")
                            row = slot.iloc[0] if not slot.empty else None
                            if row is not None and row.get('weather_text'):
                                st.caption(row['weather_text'])
                            if row is not None and pd.notna(row.get('wind_speed_ms')):
                                st.caption(f"💨 {row['wind_dir']} {row['wind_speed_ms']:.1f}m/s")
                            _precip = row.get('precipitation_mm') if row is not None else None
                            if _precip is not None and pd.notna(_precip) and float(_precip) > 0:
                                st.caption(f"🌧️ {float(_precip):.1f}mm")
                            if not hr_wx.empty and pd.notna(hr_wx.iloc[0].get('humidity')):
                                st.caption(f"湿度 {hr_wx.iloc[0]['humidity']:.0f}%")


# ============================================================
# Tab 2: 釣果分析
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _build_trend_df() -> pd.DataFrame:
    """全釣果レコードに水温トレンドカテゴリを付加して返す（1匹1行展開済み）。"""
    raw = load_data()
    if raw.empty:
        return pd.DataFrame()

    # 日別平均水温を計算してトレンドを分類
    daily_wt = (
        raw.groupby(raw['date'].dt.date)['water_temp_avg']
        .mean().reset_index()
        .rename(columns={'date': '_d', 'water_temp_avg': '_wt'})
        .sort_values('_d')
    )
    daily_wt['_wt1'] = daily_wt['_wt'].shift(1)
    daily_wt['_wt2'] = daily_wt['_wt'].shift(2)
    daily_wt['_t1'] = daily_wt['_wt'] - daily_wt['_wt1']   # 前日比
    daily_wt['_t2'] = daily_wt['_wt1'] - daily_wt['_wt2']  # 前々日→前日

    def _cat(row):
        if pd.isna(row['_t1']) or pd.isna(row['_t2']):
            return None
        t2, t1 = row['_t2'], row['_t1']
        thr = 0.3
        if   t2 < -thr and t1 < -thr: return '連続下落↘↘'
        elif t2 < -thr and t1 >= -thr: return '下落後に反発↘↗'
        elif t2 >= -thr and t1 < -thr: return '反発後に下落↗↘'
        else:                           return '連続上昇↗↗'

    daily_wt['trend_cat'] = daily_wt.apply(_cat, axis=1)
    daily_wt['_d'] = pd.to_datetime(daily_wt['_d'])

    # 展開済みデータにトレンドをマージ
    exp = load_data_exploded()
    if exp.empty:
        return pd.DataFrame()
    exp = exp.merge(daily_wt[['_d', 'trend_cat']], left_on='date', right_on='_d', how='left').drop(columns='_d')
    return exp


@st.cache_data(ttl=3600, show_spinner=False)
def _load_water_temp_pdp() -> pd.DataFrame | None:
    if not _ML_AVAILABLE or not ml_predict.models_exist():
        return None
    return ml_predict.compute_water_temp_pdp()


with tab2:
    st.subheader('釣果分析')
    _df2_base = load_data_tab2()
    _df2_base = _df2_base[
        (_df2_base['date'].dt.date >= d_from) &
        (_df2_base['date'].dt.date <= d_to) &
        (_df2_base['species'].isin(selected_species))
    ].copy()
    df2 = _df2_base[_df2_base['count'].notna() & _df2_base['water_temp_avg'].notna()].copy()

    if df2.empty:
        st.info('表示できるデータがありません。')
    else:
        # 水温帯列を追加（数値順ソート用に temp_band_num も保持）
        df2['temp_band_num'] = df2['water_temp_avg'].apply(lambda t: int(t))
        df2['temp_band']     = df2['temp_band_num'].apply(lambda t: f'{t}〜{t+1}℃')
        band_order = (df2['temp_band_num'].drop_duplicates()
                        .sort_values().apply(lambda t: f'{t}〜{t+1}℃').tolist())

        # species_detail（グレ/コッパグレ分類）で集計
        top_species = (df2.groupby('species_detail')['count'].sum()
                         .nlargest(6).index.tolist())

        # ── 1. 魚種×水温帯ヒートマップ（出現率 + 条件付き期待釣果）──
        st.markdown('##### 魚種 × 水温帯 釣れやすさ分析')
        st.caption(
            'グレは30cm未満をコッパグレ、30cm以上をグレとして分類。'
            '左: 出現率（その水温帯で釣行した日のうち実際に釣れた日の割合）'
            '　右: 釣れた時の平均匹数'
        )

        # 水温帯ごとの総釣行日数（全魚種が対象）
        _total_days = (
            df2[['date', 'temp_band_num']].drop_duplicates()
            .groupby('temp_band_num').size().rename('total_days')
        )
        # 魚種別の出現日数
        _appear = (
            df2[df2['species_detail'].isin(top_species)]
            .groupby(['species_detail', 'temp_band_num'])['date']
            .nunique().rename('appear_days').reset_index()
        )
        _appear = _appear.merge(_total_days.reset_index(), on='temp_band_num')
        _appear['presence_rate'] = (_appear['appear_days'] / _appear['total_days'] * 100).round(1)
        _appear['temp_band'] = _appear['temp_band_num'].apply(lambda t: f'{t}〜{t+1}℃')

        # 釣れた時の平均匹数（条件付き期待値）
        _cond_mean = (
            df2[df2['species_detail'].isin(top_species)]
            .groupby(['species_detail', 'temp_band_num'], as_index=False)['count']
            .mean().rename(columns={'count': 'cond_mean'})
        )
        _cond_mean['temp_band'] = _cond_mean['temp_band_num'].apply(lambda t: f'{t}〜{t+1}℃')

        heat_col1, heat_col2 = st.columns(2)

        with heat_col1:
            st.markdown('###### 出現率 (%)')
            piv_rate = (
                _appear.pivot(index='species_detail', columns='temp_band', values='presence_rate')
                .reindex(columns=band_order)
            )
            if not piv_rate.empty:
                fig_rate = px.imshow(
                    piv_rate,
                    labels=dict(x='水温帯 (℃)', y='魚種', color='出現率 (%)'),
                    color_continuous_scale='Blues',
                    zmin=0, zmax=100,
                    aspect='auto',
                    text_auto='.0f',
                )
                fig_rate.update_layout(margin=dict(t=10, b=10), height=300)
                fig_rate.update_traces(textfont_size=10)
                st.plotly_chart(fig_rate, use_container_width=True)

        with heat_col2:
            st.markdown('###### 釣れた時の平均匹数')
            piv_mean = (
                _cond_mean.pivot(index='species_detail', columns='temp_band', values='cond_mean')
                .reindex(columns=band_order)
            )
            if not piv_mean.empty:
                fig_mean = px.imshow(
                    piv_mean,
                    labels=dict(x='水温帯 (℃)', y='魚種', color='平均匹数'),
                    color_continuous_scale='YlOrRd',
                    aspect='auto',
                    text_auto='.1f',
                )
                fig_mean.update_layout(margin=dict(t=10, b=10), height=300)
                fig_mean.update_traces(textfont_size=10)
                st.plotly_chart(fig_mean, use_container_width=True)

        st.markdown('---')

        # ── 2. 水温帯別サイズ分布 & 釣れる確率 ──────────────
        col_box, col_prob = st.columns(2)

        with col_box:
            st.markdown('##### 水温帯別サイズ分布')
            sp_sel = st.multiselect(
                '魚種を選択', top_species, default=top_species[:2], key='tab2_sp'
            )
            # 1匹1行に展開したデータを使用（サイズ範囲を匹数で等分）
            df2_exp = load_data_exploded()
            df2_exp = df2_exp[
                (df2_exp['date'].dt.date >= d_from) &
                (df2_exp['date'].dt.date <= d_to) &
                (df2_exp['species'].isin(selected_species)) &
                df2_exp['water_temp_avg'].notna() &
                df2_exp['size_cm'].notna()
            ].copy()
            if not df2_exp.empty:
                df2_exp['temp_band_num'] = df2_exp['water_temp_avg'].apply(lambda t: int(t))
                df2_exp['temp_band']     = df2_exp['temp_band_num'].apply(lambda t: f'{t}〜{t+1}℃')

            box_src = df2_exp[df2_exp['species_detail'].isin(sp_sel)] if not df2_exp.empty else pd.DataFrame()
            if box_src.empty:
                st.info('対象データがありません。')
            else:
                box_src = box_src.sort_values('temp_band_num')
                fig_box = px.box(
                    box_src,
                    x='temp_band', y='size_cm', color='species_detail',
                    category_orders={'temp_band': band_order},
                    labels={'temp_band': '水温帯 (℃)', 'size_cm': 'サイズ (cm)',
                            'species_detail': '魚種'},
                    points=False,
                )
                fig_box.update_layout(margin=dict(t=10, b=10),
                                      legend=dict(orientation='h', y=-0.25))
                st.plotly_chart(fig_box, use_container_width=True)

        with col_prob:
            st.markdown('##### 水温帯別 釣れる確率（5匹以上）')
            prob_src = (
                df2[df2['species_detail'].isin(top_species)]
                .groupby(['species_detail', 'temp_band', 'temp_band_num'])
                .agg(n=('count', 'size'), hit=('count', lambda x: (x >= 5).sum()))
                .reset_index()
            )
            prob_src = prob_src[prob_src['n'] >= 3]  # サンプル数3件以上のみ
            prob_src['prob'] = prob_src['hit'] / prob_src['n']
            prob_src = prob_src.sort_values('temp_band_num')

            if prob_src.empty:
                st.info('データが不足しています。')
            else:
                fig_prob = px.line(
                    prob_src,
                    x='temp_band', y='prob', color='species_detail',
                    markers=True,
                    category_orders={'temp_band': band_order},
                    labels={'temp_band': '水温帯 (℃)', 'prob': '確率',
                            'species_detail': '魚種'},
                )
                fig_prob.update_yaxes(tickformat='.0%', range=[0, 1])
                fig_prob.update_layout(margin=dict(t=10, b=10),
                                       legend=dict(orientation='h', y=-0.25))
                st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown('---')

        # ── 3. AIモデル PDP：水温が釣果に与える影響 ──────────
        st.markdown('##### 🤖 AIモデルが示す水温の影響（Partial Dependence Plot）')
        st.caption('他の条件（潮汐・風・季節など）を実データの平均に固定した上で、'
                   '水温だけを変化させたときのモデル予測値の変化を示します。')

        with st.spinner('PDP計算中...'):
            pdp_df = _load_water_temp_pdp()

        if pdp_df is None:
            st.info('AI予測データがありません。')
        else:
            pdp_col1, pdp_col2 = st.columns(2)

            with pdp_col1:
                if 'expected_count' in pdp_df.columns:
                    fig_pdp_a = px.line(
                        pdp_df, x='water_temp', y='expected_count',
                        labels={'water_temp': '水温 (℃)', 'expected_count': '期待釣果数（匹）'},
                        line_shape='spline',
                    )
                    fig_pdp_a.update_traces(line=dict(color='tomato', width=2))
                    fig_pdp_a.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig_pdp_a, use_container_width=True)

            with pdp_col2:
                if 'go_proba' in pdp_df.columns:
                    fig_pdp_b = px.line(
                        pdp_df, x='water_temp', y='go_proba',
                        labels={'water_temp': '水温 (℃)', 'go_proba': '釣行推奨確率'},
                        line_shape='spline',
                    )
                    fig_pdp_b.update_traces(line=dict(color='steelblue', width=2))
                    fig_pdp_b.update_yaxes(tickformat='.0%', range=[0, 1])
                    fig_pdp_b.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig_pdp_b, use_container_width=True)

        # ── 生データ散布図（参考） ────────────────────────────
        with st.expander('生データ散布図（参考）', expanded=False):
            fig_scatter = px.scatter(
                df2, x='water_temp_avg', y='count', color='species_detail',
                hover_data=['date', 'spot', 'angler'],
                labels={'water_temp_avg': '水温平均 (℃)', 'count': '釣果数（匹）',
                        'species_detail': '魚種'},
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown('---')

        # ── 4. 魚種別 水温×サイズ・匹数 詳細分析 ──────────────
        st.markdown('##### 🔍 魚種別 水温×サイズ・匹数 詳細分析')
        st.caption(
            '特定の魚種を選択し、水温帯ごとのサイズ傾向・匹数・水温トレンド別の釣果を比較します。'
        )

        _trend_exp = _build_trend_df()

        # 対象魚種の選択肢（サイズデータがある魚種のみ）
        if not _trend_exp.empty and 'size_cm' in _trend_exp.columns:
            _sp_options = (
                _trend_exp[_trend_exp['size_cm'].notna()]
                .groupby('species_detail')['size_cm'].count()
                .loc[lambda s: s >= 20]
                .sort_values(ascending=False)
                .index.tolist()
            )
        else:
            _sp_options = []

        if not _sp_options:
            st.info('サイズデータが不足しています。')
        else:
            _sel_sp = st.selectbox(
                '分析する魚種を選択', _sp_options, key='tab2_detail_sp'
            )

            _sp_exp = _trend_exp[
                (_trend_exp['species_detail'] == _sel_sp) &
                _trend_exp['water_temp_avg'].notna()
            ].copy()

            if _sp_exp.empty:
                st.info('データがありません。')
            else:
                _sp_exp['temp_band_num'] = _sp_exp['water_temp_avg'].apply(
                    lambda t: int(t) if pd.notna(t) else np.nan
                )
                _sp_exp = _sp_exp[_sp_exp['temp_band_num'].notna()].copy()
                _sp_exp['temp_band_num'] = _sp_exp['temp_band_num'].astype(int)
                _sp_exp['temp_band'] = _sp_exp['temp_band_num'].apply(lambda t: f'{t}〜{t+1}℃')
                _band_order_detail = (
                    _sp_exp['temp_band_num'].drop_duplicates()
                    .sort_values().apply(lambda t: f'{t}〜{t+1}℃').tolist()
                )

                detail_col1, detail_col2 = st.columns(2)

                # 左：水温帯別サイズ推移
                with detail_col1:
                    st.markdown('###### 水温帯別 サイズ推移')
                    _size_by_temp = (
                        _sp_exp[_sp_exp['size_cm'].notna()]
                        .groupby('temp_band_num')['size_cm']
                        .agg(mean='mean', std='std', count='count')
                        .reset_index()
                    )
                    _size_by_temp = _size_by_temp[_size_by_temp['count'] >= 5].sort_values('temp_band_num')
                    _size_by_temp['temp_band'] = _size_by_temp['temp_band_num'].apply(lambda t: f'{t}〜{t+1}℃')
                    _size_by_temp['std'] = _size_by_temp['std'].fillna(0)
                    _size_by_temp['upper'] = _size_by_temp['mean'] + _size_by_temp['std']
                    _size_by_temp['lower'] = (_size_by_temp['mean'] - _size_by_temp['std']).clip(lower=0)

                    if _size_by_temp.empty:
                        st.info('サイズデータが不足しています。')
                    else:
                        import plotly.graph_objects as go
                        _fig_sz = go.Figure()
                        _fig_sz.add_trace(go.Scatter(
                            x=_size_by_temp['temp_band'], y=_size_by_temp['upper'],
                            mode='lines', line=dict(width=0), showlegend=False,
                            hoverinfo='skip',
                        ))
                        _fig_sz.add_trace(go.Scatter(
                            x=_size_by_temp['temp_band'], y=_size_by_temp['lower'],
                            fill='tonexty', mode='lines', line=dict(width=0),
                            fillcolor='rgba(255,99,71,0.2)', name='±1SD',
                            hoverinfo='skip',
                        ))
                        _fig_sz.add_trace(go.Scatter(
                            x=_size_by_temp['temp_band'], y=_size_by_temp['mean'],
                            mode='lines+markers',
                            line=dict(color='tomato', width=2),
                            marker=dict(size=7),
                            name='平均サイズ',
                            text=_size_by_temp['count'].apply(lambda n: f'n={n}'),
                            hovertemplate='%{x}<br>平均: %{y:.1f}cm<br>%{text}',
                        ))
                        _fig_sz.update_layout(
                            xaxis_title='水温帯 (℃)', yaxis_title='サイズ (cm)',
                            margin=dict(t=10, b=10),
                            legend=dict(orientation='h', y=-0.25),
                            xaxis=dict(categoryorder='array', categoryarray=_band_order_detail),
                        )
                        st.plotly_chart(_fig_sz, use_container_width=True)

                        # 統計サマリー
                        _lo_temp = _size_by_temp.iloc[0]
                        _hi_temp = _size_by_temp.iloc[-1]
                        if len(_size_by_temp) >= 3:
                            _diff = _lo_temp['mean'] - _hi_temp['mean']
                            if abs(_diff) >= 1.0:
                                _direction = '大きい' if _diff > 0 else '小さい'
                                st.caption(
                                    f"📌 {_lo_temp['temp_band']}（最低水温帯）は"
                                    f"{_hi_temp['temp_band']}（最高水温帯）より"
                                    f"平均 **{abs(_diff):.1f}cm {_direction}**"
                                    f"（{_lo_temp['mean']:.1f}cm vs {_hi_temp['mean']:.1f}cm）"
                                )

                # 右：水温帯別 匹数・釣れやすさ
                with detail_col2:
                    st.markdown('###### 水温帯別 平均匹数・5匹以上率')
                    # ここは1匹展開データではなく集計単位(レコード)が必要なのでraw使用
                    _raw_sp = load_data_tab2()
                    _raw_sp = _raw_sp[
                        (_raw_sp['species_detail'] == _sel_sp) &
                        _raw_sp['water_temp_avg'].notna() &
                        _raw_sp['count'].notna()
                    ].copy()
                    if not _raw_sp.empty:
                        _raw_sp['temp_band_num'] = _raw_sp['water_temp_avg'].apply(lambda t: int(t))
                        _raw_sp['temp_band'] = _raw_sp['temp_band_num'].apply(lambda t: f'{t}〜{t+1}℃')
                        _cnt_by_temp = (
                            _raw_sp.groupby('temp_band_num')
                            .agg(avg_count=('count', 'mean'), hit_rate=('count', lambda x: (x >= 5).mean()), n=('count', 'size'))
                            .reset_index()
                        )
                        _cnt_by_temp = _cnt_by_temp[_cnt_by_temp['n'] >= 3].sort_values('temp_band_num')
                        _cnt_by_temp['temp_band'] = _cnt_by_temp['temp_band_num'].apply(lambda t: f'{t}〜{t+1}℃')

                        if not _cnt_by_temp.empty:
                            _fig_cnt = go.Figure()
                            _fig_cnt.add_trace(go.Bar(
                                x=_cnt_by_temp['temp_band'], y=_cnt_by_temp['avg_count'],
                                name='平均匹数', marker_color='steelblue',
                                yaxis='y', opacity=0.75,
                                text=_cnt_by_temp['n'].apply(lambda n: f'n={n}'),
                                textposition='outside',
                            ))
                            _fig_cnt.add_trace(go.Scatter(
                                x=_cnt_by_temp['temp_band'], y=_cnt_by_temp['hit_rate'],
                                name='5匹以上率', mode='lines+markers',
                                line=dict(color='darkorange', width=2),
                                marker=dict(size=7),
                                yaxis='y2',
                            ))
                            _fig_cnt.update_layout(
                                xaxis=dict(categoryorder='array', categoryarray=_band_order_detail),
                                yaxis=dict(title='平均匹数'),
                                yaxis2=dict(title='5匹以上率', overlaying='y', side='right',
                                            tickformat='.0%', range=[0, 1]),
                                margin=dict(t=10, b=10),
                                legend=dict(orientation='h', y=-0.25),
                            )
                            st.plotly_chart(_fig_cnt, use_container_width=True)

                st.markdown('')
                # 水温トレンド別 サイズ＋匹数比較
                st.markdown('###### 水温トレンド別 サイズ・匹数比較')
                st.caption('前日・前々日の水温変化パターンが釣果サイズ・匹数にどう影響するか。')

                _TREND_ORDER = ['連続下落↘↘', '下落後に反発↘↗', '反発後に下落↗↘', '連続上昇↗↗']

                # サイズ（展開済み）
                _trend_sz = (
                    _sp_exp[_sp_exp['trend_cat'].notna() & _sp_exp['size_cm'].notna()]
                    .groupby('trend_cat')['size_cm']
                    .agg(mean='mean', count='count')
                    .reset_index()
                )
                # 匹数（rawレコード）
                if not _raw_sp.empty:
                    _trend_cnt = (
                        _raw_sp[_raw_sp['trend_cat'].notna() & _raw_sp['count'].notna()]
                        .groupby('trend_cat')['count']
                        .agg(avg_count='mean', n='size')
                        .reset_index()
                    ) if 'trend_cat' in _raw_sp.columns else pd.DataFrame()
                else:
                    _trend_cnt = pd.DataFrame()

                # trend_cat を _raw_sp にマージ（未付与の場合）
                if not _raw_sp.empty and 'trend_cat' not in _raw_sp.columns:
                    _raw_sp2 = _raw_sp.merge(
                        _trend_exp[['date', 'trend_cat']].drop_duplicates('date'),
                        on='date', how='left'
                    )
                    _trend_cnt = (
                        _raw_sp2[_raw_sp2['trend_cat'].notna() & _raw_sp2['count'].notna()]
                        .groupby('trend_cat')['count']
                        .agg(avg_count='mean', n='size')
                        .reset_index()
                    )

                if _trend_sz.empty and _trend_cnt.empty:
                    st.info('トレンドデータが不足しています。')
                else:
                    trend_col1, trend_col2 = st.columns(2)

                    with trend_col1:
                        if not _trend_sz.empty and _trend_sz['count'].sum() >= 10:
                            _trend_sz_filt = _trend_sz[_trend_sz['count'] >= 5]
                            _fig_tsz = px.bar(
                                _trend_sz_filt,
                                x='trend_cat', y='mean',
                                category_orders={'trend_cat': _TREND_ORDER},
                                labels={'trend_cat': '水温トレンド', 'mean': '平均サイズ (cm)'},
                                color='mean',
                                color_continuous_scale='RdYlGn',
                                text=_trend_sz_filt['count'].apply(lambda n: f'n={n}'),
                            )
                            _fig_tsz.update_traces(textposition='outside')
                            _fig_tsz.update_layout(margin=dict(t=10, b=10),
                                                   showlegend=False,
                                                   coloraxis_showscale=False)
                            st.markdown('平均サイズ (cm)')
                            st.plotly_chart(_fig_tsz, use_container_width=True)
                        else:
                            st.info('サイズデータが不足しています。')

                    with trend_col2:
                        if not _trend_cnt.empty and _trend_cnt['n'].sum() >= 10:
                            _trend_cnt_filt = _trend_cnt[_trend_cnt['n'] >= 3]
                            _fig_tcnt = px.bar(
                                _trend_cnt_filt,
                                x='trend_cat', y='avg_count',
                                category_orders={'trend_cat': _TREND_ORDER},
                                labels={'trend_cat': '水温トレンド', 'avg_count': '平均匹数'},
                                color='avg_count',
                                color_continuous_scale='Blues',
                                text=_trend_cnt_filt['n'].apply(lambda n: f'n={n}'),
                            )
                            _fig_tcnt.update_traces(textposition='outside')
                            _fig_tcnt.update_layout(margin=dict(t=10, b=10),
                                                    showlegend=False,
                                                    coloraxis_showscale=False)
                            st.markdown('平均匹数')
                            st.plotly_chart(_fig_tcnt, use_container_width=True)
                        else:
                            st.info('匹数データが不足しています。')


# ============================================================
# Tab 3: 釣り場ランキング
# ============================================================
with tab3:
    st.subheader('釣り場ランキング（上位20磯）')
    df3 = df[df['spot'].notna() & df['count'].notna()]

    if df3.empty:
        st.info('表示できるデータがありません。')
    else:
        agg = df3.groupby('spot').agg(
            total_count=('count', 'sum'),
            avg_size=('size_max_cm', 'mean'),
            max_size=('size_max_cm', 'max'),
            species_list=('species', lambda x: '・'.join(sorted(x.dropna().unique()))),
        ).reset_index().sort_values('total_count', ascending=False).head(20)

        fig_rank = px.bar(
            agg.sort_values('total_count'),
            x='total_count', y='spot', orientation='h',
            labels={'total_count': '総釣果数（匹）', 'spot': '釣り場'},
            title='釣り場別 総釣果数',
            hover_data=['avg_size', 'max_size', 'species_list'],
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        agg_display = agg.rename(columns={
            'spot': '釣り場', 'total_count': '総釣果数',
            'avg_size': '平均サイズ(cm)', 'max_size': '最大サイズ(cm)',
            'species_list': '魚種',
        })
        agg_display[['平均サイズ(cm)', '最大サイズ(cm)']] = (
            agg_display[['平均サイズ(cm)', '最大サイズ(cm)']].round(1)
        )
        st.dataframe(agg_display, use_container_width=True, hide_index=True)


# ============================================================
# Tab 4: 月別・季節別
# ============================================================
with tab4:
    st.subheader('月別・季節別の釣果傾向')
    df4 = df[df['species'].notna() & df['count'].notna()]

    if df4.empty:
        st.info('表示できるデータがありません。')
    else:
        col1, col2 = st.columns(2)

        with col1:
            # 月×魚種ヒートマップ（魚種ごとの月別割合）
            heat_df = df4.groupby(['month', 'species'])['count'].sum().reset_index()
            heat_pivot = heat_df.pivot(index='species', columns='month', values='count').fillna(0)
            # 各魚種の合計で割って月別割合（%）に正規化
            species_total = heat_pivot.sum(axis=1)
            heat_pivot = heat_pivot.div(species_total, axis=0).mul(100).round(1)
            heat_pivot.columns = [f'{m}月' for m in heat_pivot.columns]
            fig_heat = px.imshow(
                heat_pivot,
                labels=dict(x='月', y='魚種', color='割合 (%)'),
                title='月×魚種 釣果割合ヒートマップ（魚種ごと）',
                color_continuous_scale='Blues',
                zmin=0, zmax=100,
                text_auto='.0f',
            )
            fig_heat.update_traces(textfont_size=9)
            fig_heat.update_layout(margin=dict(t=40, b=10))
            st.caption('各行（魚種）の月別割合。全月合計が100%になるよう正規化。色が濃い月ほどその魚種の釣果が集中している。')
            st.plotly_chart(fig_heat, use_container_width=True)

        with col2:
            # 季節ごとの魚種構成円グラフ
            season_order = ['春', '夏', '秋', '冬']
            available_seasons = [s for s in season_order if s in df4['season'].values]
            selected_season = st.selectbox('季節を選択', available_seasons)
            season_df = df4[df4['season'] == selected_season]
            pie_df = season_df.groupby('species')['count'].sum().reset_index()
            fig_pie = px.pie(
                pie_df, values='count', names='species',
                title=f'{selected_season}の魚種構成',
            )
            st.plotly_chart(fig_pie, use_container_width=True)


# ============================================================
# Tab 5: 波高・天候条件
# ============================================================
with tab5:
    st.subheader('波高・天候条件と釣果')
    df5 = df[df['count'].notna()]

    if df5.empty:
        st.info('表示できるデータがありません。')
    else:
        col1, col2 = st.columns(2)

        with col1:
            # 波高帯別の釣果数 箱ひげ図
            df5_wave = df5[df5['wave_height_m'].notna()].copy()
            df5_wave['wave_band'] = (df5_wave['wave_height_m']
                                     .apply(lambda w: f'{int(w)}〜{int(w)+1}m'))
            if not df5_wave.empty:
                fig_box = px.box(
                    df5_wave.sort_values('wave_height_m'),
                    x='wave_band', y='count',
                    labels={'wave_band': '波高帯', 'count': '釣果数（匹）'},
                    title='波高帯別の釣果数分布',
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info('波高データがありません。')

        with col2:
            # 天候×波高×水温 バブルチャート
            df5_bubble = df5[
                df5['wave_height_m'].notna() &
                df5['water_temp_avg'].notna() &
                df5['weather'].notna()
            ]
            if not df5_bubble.empty:
                fig_bubble = px.scatter(
                    df5_bubble,
                    x='water_temp_avg', y='wave_height_m',
                    size='count', color='weather',
                    hover_data=['date', 'spot', 'species'],
                    labels={
                        'water_temp_avg': '水温平均 (℃)',
                        'wave_height_m':  '波高 (m)',
                        'weather':        '天候',
                        'count':          '釣果数',
                    },
                    title='天候 × 波高 × 水温 バブルチャート',
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
            else:
                st.info('表示に必要なデータが不足しています。')


