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

/* ライトモード固定（ダークモード無効化） */
:root { color-scheme: light; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Hiragino Sans', 'Yu Gothic UI', sans-serif;
}

/* ── 背景・テキスト色（ライト固定） ── */
.stApp {
    background-color: #EDF2F7 !important;
    color: #1C3448 !important;
}
.main .block-container { padding-top: 1.5rem; max-width: 1400px; }

/* メインエリアの全テキストを暗色に統一 */
.main p, .main li, .main span:not([data-baseweb]),
.main label, .main div[data-testid="stText"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #1C3448 !important;
}

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
[data-testid="stDataFrame"] * { color: #1C3448 !important; }

/* スクロール可能テーブルの右端にフェードを表示 */
[data-testid="stDataFrame"] > div {
    position: relative;
}
[data-testid="stDataFrame"] > div::after {
    content: '▶';
    position: absolute;
    top: 50%;
    right: 4px;
    transform: translateY(-50%);
    font-size: 10px;
    color: #1B8FA8;
    pointer-events: none;
    z-index: 10;
    background: linear-gradient(to right, transparent, rgba(237,242,247,0.95) 40%);
    padding: 2px 4px 2px 16px;
}

/* ── Metricカード ── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 4px solid #1B8FA8;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"],
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1C3448 !important;
}


/* ── ボタン（サイドバーのみグラデーション） ── */
[data-testid="stSidebar"] .stButton > button {
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
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(11,61,92,0.35) !important;
}
[data-testid="stSidebar"] .stButton > button:active { transform: translateY(0) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    border: 1px solid #D9E8F2 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
    margin-bottom: 8px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 500;
    color: #0B3D5C !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #F0F7FB !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] li,
[data-testid="stExpander"] span:not([data-baseweb]) {
    color: #1C3448 !important;
}

/* ── Statusボックス ── */
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
}

/* ── Alertボックス ── */
[data-testid="stAlert"] { border-radius: 10px !important; }
[data-testid="stAlert"] p { color: inherit !important; }

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

/* ══════════════════════════════════════════════════
   日別判断カード — HTMLカードデザイン
   ══════════════════════════════════════════════════ */
.dcard-wrap {
    display: flex;
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    overflow: hidden;
    height: 80px;
    pointer-events: none;
    user-select: none;
}
.dcard-wrap.drm-today { box-shadow: 0 3px 18px rgba(0,0,0,0.15); }

/* 左カラム: 星・釣果数・日付 */
.dcard-l {
    width: 68px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #F2F7FA;
    border-right: 1px solid #DDE9F2;
    padding: 6px 4px;
    gap: 3px;
}
.dcard-stars { font-size: 0.68rem; color: #D4A017; letter-spacing: -1px; line-height: 1; }
.dcard-count { font-size: 0.9rem; font-weight: 700; color: #0B3D5C; line-height: 1; }
.dcard-date  { font-size: 0.72rem; text-align: center; color: #2C4E6A; font-weight: 600; line-height: 1.3; }
.dcard-wd    { font-size: 0.62rem; color: #6A7E8A; font-weight: 400; }
.dcard-today-badge {
    font-size: 0.58rem; background: #E07B00; color: #fff;
    border-radius: 3px; padding: 1px 4px; font-weight: 700; line-height: 1.5;
}

/* 右エリア */
.dcard-r { flex: 1; display: flex; flex-direction: column; min-width: 0; }

/* 上段: 潮型・気温 */
.dcard-top {
    padding: 8px 12px 5px;
    font-size: 0.88rem; font-weight: 700; color: #0B3D5C;
    border-bottom: 1px solid #EEF3F7;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    line-height: 1.2;
}

/* 下段: 条件＋出船 */
.dcard-bot { display: flex; align-items: stretch; flex: 1; overflow: hidden; }
.dcard-conds {
    flex: 1; display: flex; flex-direction: column; justify-content: center;
    padding: 4px 10px; gap: 2px; min-width: 0; overflow: hidden;
}
.dcard-conds-r1 {
    font-size: 0.74rem; font-weight: 500; color: #2C4E6A;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.dcard-conds-r2 {
    font-size: 0.66rem; color: #5A6E7D;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* 出船: 縦書きストリップ */
.dcard-dep {
    writing-mode: vertical-rl; text-orientation: mixed;
    font-size: 0.68rem; font-weight: 700; color: #fff;
    padding: 4px 3px; min-width: 22px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: 0.1em;
}
.dcard-wrap.drm-go    .dcard-dep { background: #1a7a4a; }
.dcard-wrap.drm-check .dcard-dep { background: #b07d00; }
.dcard-wrap.drm-stop  .dcard-dep { background: #c0392b; }

/* marker spanは非表示（:has()はdisplay:noneでもマッチする） */
.dcard-marker { display: none !important; }
/* カード間スペーサー: flex gap 1つ分を稼ぐためだけの空要素 */
.dcard-spacer { height: 0; margin: 0; padding: 0; line-height: 0; font-size: 0; }
[data-testid="element-container"]:has(.dcard-spacer) {
    margin: 0 !important; padding: 0 !important; min-height: 0 !important;
}
[data-testid="element-container"]:has(.dcard-spacer) > div {
    margin: 0 !important; padding: 0 !important;
}

/* カードコンテナ: 内部余白を完全除去 */
[data-testid="element-container"]:has(.dcard-marker) {
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="element-container"]:has(.dcard-marker) > div,
[data-testid="element-container"]:has(.dcard-marker) .stMarkdown,
[data-testid="element-container"]:has(.dcard-marker) .stMarkdown > div {
    margin: 0 !important;
    padding: 0 !important;
}
/* カード: margin-bottom負値でスペースを相殺 */
.dcard-wrap {
    position: relative !important;
    z-index: 1 !important;
    margin-bottom: -80px !important;
}

/* 透明オーバーレイボタン（JSで.dcard-btn-wrapと.dcard-btnを付与） */
.dcard-btn-wrap {
    position: relative !important;
    z-index: 2 !important;
    margin-bottom: 8px !important;
    padding: 0 !important;
}
.dcard-btn-wrap .stButton {
    margin: 0 !important;
    padding: 0 !important;
}
button.dcard-btn,
button.dcard-btn:hover,
button.dcard-btn:active,
button.dcard-btn:focus,
button.dcard-btn:focus-visible {
    height: 80px !important;
    background: transparent !important;
    background-image: none !important;
    border: none !important;
    box-shadow: none !important;
    color: transparent !important;
    cursor: pointer !important;
    width: 100% !important;
    padding: 0 !important;
    border-radius: 12px !important;
    transform: none !important;
    font-size: 0 !important;
    min-height: 0 !important;
    letter-spacing: 0 !important;
    outline: none !important;
}
button.dcard-btn:hover {
    background: rgba(0,0,0,0.04) !important;
}

/* ── 地点セグメントコントロール ── */
[data-testid="stRadio"] > div {
    background: #EEF5FA;
    border-radius: 10px;
    padding: 4px 6px;
    border: 1.5px solid #A8C8DC;
    display: inline-flex;
    gap: 4px;
    flex-wrap: nowrap;
}
[data-testid="stRadio"] label {
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 4px 14px !important;
    border-radius: 7px !important;
    cursor: pointer !important;
    color: #4A7A95 !important;
}

/* ── タブ：モバイルで絵文字なしショート表示 ── */
@media (max-width: 480px) {
    .stTabs [data-baseweb="tab"] {
        padding: 6px 8px !important;
        font-size: 0.74rem !important;
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

_kpi_maxsize_str = f'{_kpi_maxsize:.0f}' if pd.notna(_kpi_maxsize) else '―'
_CARD = (
    "background:#fff;border-radius:12px;"
    "padding:clamp(12px,3.5vw,18px) clamp(14px,4vw,20px);"
    "box-shadow:0 2px 10px rgba(0,0,0,0.08);"
)
_CARD_L = _CARD + "border-left:4px solid #1B8FA8;"
_CARD_R = _CARD + "border-left:4px solid #22AECB;"
_LBL = "font-size:clamp(0.6rem,2vw,0.72rem);color:#6B7B8D;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;"
_VAL = "font-size:clamp(1.05rem,4.5vw,1.45rem);color:#0B3D5C;font-weight:700;line-height:1.2;"
_UNIT = "font-size:0.65em;font-weight:500;margin-left:3px;color:#4A6070;"
st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:clamp(8px,2.5vw,14px);margin-bottom:20px;">
  <div style="{_CARD_L}"><div style="{_LBL}">総釣果数</div>
    <div style="{_VAL}">{_kpi_total:,}<span style="{_UNIT}">匹</span></div></div>
  <div style="{_CARD_L}"><div style="{_LBL}">釣行日数</div>
    <div style="{_VAL}">{_kpi_days:,}<span style="{_UNIT}">日</span></div></div>
  <div style="{_CARD_R}"><div style="{_LBL}">釣り場数</div>
    <div style="{_VAL}">{_kpi_spots}<span style="{_UNIT}">磯</span></div></div>
  <div style="{_CARD_R}"><div style="{_LBL}">最大サイズ</div>
    <div style="{_VAL}">{_kpi_maxsize_str}<span style="{_UNIT}">cm</span></div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="margin-bottom:8px;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------

tab0, tab1, tab2 = st.tabs([
    '🎣 釣行判断',
    '📊 分析',
    '🏆 磯ランキング',
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
# Tab 0: 今日の判断 — 詳細ダイアログ
# ============================================================

@st.dialog('📅 釣行詳細', width='large')
def _show_day_detail(sel_date, sel_pred, mw_row, h_wx, h_fc, h_tide, h_br, today, wdays,
                     load_ranking_fn, build_reasons_fn, species_lift_fn,
                     hourly_df=None):
    """選択した日の詳細情報をダイアログで表示する。"""
    import streamlit.components.v1 as _stc
    _stc.html(
        '<script>(function(){'
        'var SEL="[data-testid=\\"stDialogScrollArea\\"]";'
        'var T0=Date.now();var DUR=5000;'
        'function rst(){'
        '[window.parent,window.top].forEach(function(w){'
        'try{w.document.querySelectorAll(SEL).forEach(function(e){e.scrollTop=0;});}catch(e){}});}'
        '[80,250,600,1200,2500].forEach(function(t){setTimeout(rst,t)});'
        '[window.parent,window.top].forEach(function(w){'
        'try{'
        'var ob=new MutationObserver(function(){'
        'if(Date.now()-T0<DUR){rst();}else{ob.disconnect();}});'
        'ob.observe(w.document.body,{childList:true,subtree:true});'
        '}catch(e){}});'
        '})();</script>',
        height=1, scrolling=False,
    )
    _d = sel_date
    _wd = wdays[_d.weekday()]
    _is_today = (_d == today)

    # ── GO/CHECK/STOP バナー ──────────────────────────────────
    _rp   = mw_row['risk_prob'] if mw_row is not None else 0.0
    _spd  = mw_row['wind_max_ms'] if mw_row is not None else None
    _gp   = sel_pred.get('go_proba') if sel_pred else None
    _ec   = sel_pred.get('expected_count') if sel_pred else None
    _sp_str = species_lift_fn(sel_pred.get('species_proba', {}) if sel_pred else {}, h_br, top_n=2)

    if _rp >= 0.90:
        _verdict, _vc, _vbg = '✖ STOP', '#7a1c24', '#f8d7da'
        _vmsg = '出船困難な可能性があります'
    elif _rp >= 0.75:
        _verdict, _vc, _vbg = '⚠️ CHECK', '#7a5f00', '#fff3cd'
        _vmsg = '出船できない可能性があります。確認してください'
    else:
        _verdict, _vc, _vbg = '✅ GO', '#1a7a4a', '#d4edda'
        _vmsg = '出船できる見込みです'

    _today_badge = ' <span style="background:#1B8FA8;color:#fff;border-radius:6px;padding:2px 8px;font-size:0.75rem;vertical-align:middle;">今日</span>' if _is_today else ''
    st.markdown(
        f'<h3 style="margin:0 0 12px;">{_d.month}/{_d.day}（{_wd}）{_today_badge}</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
<div style="background:{_vbg};border-left:6px solid {_vc};
     padding:18px 22px;border-radius:10px;margin-bottom:18px;">
  <div style="font-size:1.8rem;font-weight:800;color:{_vc};line-height:1.1;">{_verdict}</div>
  <div style="font-size:0.9rem;color:{_vc};margin-top:4px;">{_vmsg}</div>
  <div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:20px;color:#333;font-size:0.88rem;">
    {'<span>推奨スコア&nbsp;<b>' + f'{_gp*100:.0f}%</b></span>' if _gp is not None else ''}
    {'<span>期待釣果&nbsp;<b>' + f'{_ec:.1f}&nbsp;匹</b></span>' if _ec is not None else ''}
    {'<span>注目魚種&nbsp;<b>' + _sp_str + '</b></span>' if _sp_str else ''}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 4条件カード ────────────────────────────────────────────
    _wx_row = None
    if h_wx is not None:
        _wr = h_wx[h_wx['date'] == _d]
        _wx_row = _wr.iloc[0] if not _wr.empty else None
    _fc_row = None
    if h_fc is not None:
        _fr = h_fc[h_fc['date'] == _d]
        _fc_row = _fr.iloc[0] if not _fr.empty else None
    _tide_name = None
    if h_tide is not None and not h_tide.empty:
        _tn = h_tide[(h_tide['date'] == _d) & h_tide['tide_name'].notna()]
        if not _tn.empty:
            _tide_name = str(_tn.iloc[0]['tide_name'])

    _weather_raw = (sel_pred.get('weather') if sel_pred else None)
    if not _weather_raw and _wx_row is not None and pd.notna(_wx_row.get('weather')):
        _weather_raw = str(_wx_row['weather'])
    _weather_str = _weather_raw or '–'

    _wave_val = (float(_fc_row['forecast_wave_height_m'])
                 if _fc_row is not None and pd.notna(_fc_row.get('forecast_wave_height_m')) else None)
    _wave_str = f"{_wave_val:.1f} m" if _wave_val is not None else '–'

    _wind_ms_val = None
    if sel_pred and sel_pred.get('wind_ms_max') is not None:
        try:
            _wind_ms_val = float(sel_pred['wind_ms_max']) / 3.6
        except (TypeError, ValueError):
            pass

    _recent_wt = df_all[
        df_all['date'] >= (df_all['date'].max() - pd.Timedelta(days=7))
    ]['water_temp_avg'].mean()
    _wt_str = f"{_recent_wt:.1f} ℃" if pd.notna(_recent_wt) else '–'
    _tide_str = _tide_name or (sel_pred.get('tide_name') if sel_pred else None) or '–'

    _dc = ("background:#fff;border-radius:12px;"
           "padding:clamp(12px,3.5vw,18px) clamp(14px,4vw,20px);"
           "box-shadow:0 2px 10px rgba(0,0,0,0.08);")
    _dcL = _dc + "border-left:4px solid #1B8FA8;"
    _dcR = _dc + "border-left:4px solid #22AECB;"
    _dlb = "font-size:clamp(0.6rem,2vw,0.72rem);color:#6B7B8D;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;"
    _dvl = "font-size:clamp(1.0rem,4vw,1.3rem);color:#0B3D5C;font-weight:700;line-height:1.2;"
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:clamp(8px,2.5vw,12px);margin-bottom:18px;">
  <div style="{_dcL}"><div style="{_dlb}">☁️ 天気</div><div style="{_dvl}">{_weather_str}</div></div>
  <div style="{_dcL}"><div style="{_dlb}">🌊 波高（予報）</div><div style="{_dvl}">{_wave_str}</div></div>
  <div style="{_dcR}"><div style="{_dlb}">🌡️ 水温（直近7日）</div><div style="{_dvl}">{_wt_str}</div></div>
  <div style="{_dcR}"><div style="{_dlb}">🌙 潮</div><div style="{_dvl}">{_tide_str}</div></div>
</div>
""", unsafe_allow_html=True)

    # ── 6時間ごとの詳細グリッド ────────────────────────────────
    _hd = hourly_df[hourly_df['date'] == _d] if hourly_df is not None and not hourly_df.empty else pd.DataFrame()
    if not _hd.empty:
        _HOURS_D = [0, 6, 12, 18]
        _XRANGE_D = [-3, 21]
        _XTICKS_D = dict(tickvals=[0, 6, 12, 18], ticktext=['0:00', '6:00', '12:00', '18:00'])
        _CM_D = dict(l=45, r=10, t=8, b=8)
        _tc_d = "padding:4px 8px;text-align:center;font-size:0.76rem;color:#1C3448;"
        _tl_d = "padding:4px 6px;color:#888;font-weight:600;font-size:0.76rem;white-space:nowrap;"
        _th_d = "padding:5px 8px;text-align:center;font-size:0.76rem;font-weight:700;color:#0B3D5C;border-bottom:2px solid #DDE8F5;"

        def _hv_dd(hour_, col_, fmt='{}'):
            r_ = _hd[_hd['hour'] == hour_]
            if r_.empty or not pd.notna(r_.iloc[0].get(col_)): return '–'
            return fmt.format(r_.iloc[0][col_])

        def _wxe_hd(s):
            if not s or s == '–': return s
            s = str(s)
            if '雪' in s: return '❄️'
            if '雷' in s: return '⛈️'
            if '雨' in s and '晴' in s: return '🌦️'
            if '雨' in s: return '🌧️'
            if '曇' in s and '晴' in s: return '🌤️'
            if '曇' in s: return '☁️'
            if '晴' in s: return '☀️'
            return s

        def _tr_d(label, cells_html):
            return f'<tr><td style="{_tl_d}">{label}</td>{cells_html}</tr>'

        def _tds_d(col, fmt='{}'):
            return ''.join(f'<td style="{_tc_d}">{_hv_dd(h, col, fmt)}</td>' for h in _HOURS_D)

        _hour_headers_d = ''.join(f'<th style="{_th_d}">{h:02d}:00</th>' for h in _HOURS_D)
        _wx_tds_d  = ''.join(f'<td style="{_tc_d}">{_wxe_hd(_hv_dd(h,"weather_text")) or _hv_dd(h,"weather_text")}</td>' for h in _HOURS_D)
        _wind_tds_d = ''.join(f'<td style="{_tc_d}">{_hv_dd(h,"wind_dir")}<br>{_hv_dd(h,"wind_speed_ms","{:.0f}m/s")}</td>' for h in _HOURS_D)
        st.markdown(f"""
<div style="overflow-x:auto;margin-bottom:16px;background:#F8FBFF;border-radius:10px;padding:10px 12px;">
<table style="width:100%;min-width:260px;border-collapse:collapse;">
  <thead><tr><th style="{_tl_d}"></th>{_hour_headers_d}</tr></thead>
  <tbody>
    {_tr_d('天気', _wx_tds_d)}
    {_tr_d('気温', _tds_d('temp_c', '{:.0f}℃'))}
    {_tr_d('湿度', _tds_d('humidity', '{:.0f}%'))}
    {_tr_d('風', _wind_tds_d)}
    {_tr_d('降水', _tds_d('precipitation_mm', '{:.1f}mm'))}
  </tbody>
</table>
</div>
""", unsafe_allow_html=True)

        # 気温・潮位グラフ
        _cfg_d = {'displayModeBar': False, 'scrollZoom': False, 'staticPlot': False}
        _ch_l, _ch_r = st.columns(2)
        with _ch_l:
            _temp_pts = [(h, (lambda s: s.iloc[0]['temp_c'] if not s.empty and pd.notna(s.iloc[0].get('temp_c')) else None)(_hd[_hd['hour'] == h])) for h in _HOURS_D]
            if any(v is not None for _, v in _temp_pts):
                _xs = [h for h, v in _temp_pts if v is not None]
                _ys = [v for _, v in _temp_pts if v is not None]
                _fig_t = go.Figure(go.Scatter(x=_xs, y=_ys, mode='lines+markers',
                    line=dict(shape='spline', color='tomato', width=2),
                    marker=dict(size=7, color='tomato'), showlegend=False))
                _fig_t.update_xaxes(range=_XRANGE_D, showgrid=True, gridcolor='#eee', zeroline=False, fixedrange=True, **_XTICKS_D)
                _fig_t.update_yaxes(title_text='℃', tickformat='.0f', showgrid=True, gridcolor='#eee', fixedrange=True)
                _fig_t.update_layout(height=180, margin=_CM_D, plot_bgcolor='white', paper_bgcolor='white',
                    title=dict(text='気温', font_size=13, x=0.02), dragmode=False)
                st.plotly_chart(_fig_t, use_container_width=True, config=_cfg_d)
        with _ch_r:
            if h_tide is not None and not h_tide.empty:
                _dtplot = h_tide[h_tide['date'] == _d].copy()
                if not _dtplot.empty:
                    _dtplot['height_cm'] = pd.to_numeric(_dtplot['height_cm'], errors='coerce')
                    _dtplot['hour_f'] = _dtplot['time'].apply(lambda t: int(str(t).split(':')[0]) + int(str(t).split(':')[1]) / 60)
                    _dtplot = _dtplot.sort_values('hour_f')
                    _fig_td = go.Figure(go.Scatter(
                        x=_dtplot['hour_f'], y=_dtplot['height_cm'], mode='lines+markers',
                        line=dict(shape='spline', color='steelblue', width=2),
                        marker=dict(size=7, color='steelblue'),
                        text=_dtplot.apply(lambda r: f"{r['type']} {str(r['time'])[:5]} {r['height_cm']:.0f}cm", axis=1),
                        hovertemplate='%{text}<extra></extra>', showlegend=False))
                    _fig_td.update_xaxes(range=_XRANGE_D, showgrid=True, gridcolor='#eee', fixedrange=True, **_XTICKS_D)
                    _fig_td.update_yaxes(title_text='cm', tickformat='.0f', showgrid=True, gridcolor='#eee', fixedrange=True)
                    _fig_td.update_layout(height=180, margin=_CM_D, plot_bgcolor='white', paper_bgcolor='white',
                        title=dict(text='潮位', font_size=13, x=0.02), dragmode=False)
                    st.plotly_chart(_fig_td, use_container_width=True, config=_cfg_d)

    # ── 懸念事項 / 好条件 ─────────────────────────────────────
    _rising = sel_pred.get('rising_ratio') if sel_pred else None
    _reasons = build_reasons_fn(
        pred=sel_pred,
        wind_ms=_wind_ms_val,
        wave_val=_wave_val,
        wt=_recent_wt if pd.notna(_recent_wt) else None,
        tide_name=_tide_name,
        rising=_rising,
    )
    _bad  = [(ic, tx) for ic, tx, b in _reasons if b]
    _good = [(ic, tx) for ic, tx, b in _reasons if not b]

    def _reason_cards(items, accent, bg):
        html = f'<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px;">'
        for ic, tx in items:
            html += (
                f'<div style="background:{bg};border-left:4px solid {accent};'
                f'border-radius:8px;padding:10px 14px;font-size:0.88rem;color:#1C3448;">'
                f'<span style="margin-right:6px;">{ic}</span>{tx}</div>'
            )
        html += '</div>'
        return html

    st.markdown('#### ⚠️ 懸念事項' if _bad else '#### ✅ 懸念事項なし')
    if _bad:
        st.markdown(_reason_cards(_bad, '#c0392b', '#fdf3f3'), unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#1a7a4a;font-size:0.9rem;margin-bottom:14px;">現在の予報では大きな懸念点はありません。</div>', unsafe_allow_html=True)

    st.markdown('#### ✅ 好条件の理由' if _good else '#### ℹ️ 好条件なし')
    if _good:
        st.markdown(_reason_cards(_good, '#1a7a4a', '#f0faf4'), unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#888;font-size:0.9rem;margin-bottom:14px;">この日は好条件の要素が少ない状況です。</div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin:8px 0 16px;">', unsafe_allow_html=True)

    # ── 磯ランキング TOP5 ─────────────────────────────────────
    st.markdown(f'#### 🏆 {_d.month}/{_d.day} の磯 期待度ランキング TOP5')
    with st.spinner('計算中...'):
        _dlg_rank = load_ranking_fn(_d.isoformat())
    if _dlg_rank.empty:
        st.info('磯ランキングデータがありません。')
    else:
        for _ri, _rrow in _dlg_rank.head(5).iterrows():
            _rn = _ri + 1
            _medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(_rn, f'{_rn}.')
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #eee;">
  <span style="font-size:1.1rem;min-width:28px;">{_medal}</span>
  <span style="font-weight:600;flex:1;">{_rrow['spot']}</span>
  <span style="color:#e07b39;font-weight:700;">{_rrow['expected_count']:.1f} 匹</span>
  <span style="color:#888;font-size:0.82rem;min-width:36px;text-align:right;">{_rrow['go_proba']*100:.0f}%</span>
</div>""", unsafe_allow_html=True)
    st.caption('詳細は「🏆 釣り場ランキング」タブへ')

    st.markdown('<hr style="margin:8px 0 16px;">', unsafe_allow_html=True)

    # ── 注目魚種 ──────────────────────────────────────────────
    st.markdown('#### 🐟 注目魚種（平均比）')
    if sel_pred:
        _sp_raw = sel_pred.get('species_proba', {})
        _sp_lifts = {
            sp: p / h_br[sp] if h_br.get(sp, 0) > 0 else 1.0
            for sp, p in _sp_raw.items()
        }
        for _sp, _lv in sorted(_sp_lifts.items(), key=lambda x: -x[1]):
            _sc_color = '#1a7a4a' if _lv >= 1.2 else ('#7a5f00' if _lv >= 0.8 else '#999')
            _bar_w = min(int(_lv / 2.0 * 100), 100)
            st.markdown(f"""
<div style="margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
    <span style="font-weight:600;">{_sp}</span>
    <span style="color:{_sc_color};font-weight:700;">{_lv:.2f}x</span>
  </div>
  <div style="background:#eee;border-radius:4px;height:6px;">
    <div style="background:{_sc_color};width:{_bar_w}%;height:6px;border-radius:4px;"></div>
  </div>
</div>""", unsafe_allow_html=True)
        st.caption('1.0x = 平均並み　2.0x = 平均の2倍釣れやすい')
    else:
        st.info('AIモデルが必要です。')



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


# ============================================================
# Tab 0: 釣行判断（今日 + 予測 統合）
# ============================================================
with tab0:
    _today = date.today()
    _WDAYS = '月火水木金土日'

    # 地点セレクター
    st.markdown('<p style="font-size:0.82rem;color:#4A7A95;margin-bottom:4px;font-weight:600;">地点</p>', unsafe_allow_html=True)
    _u_loc = st.radio('', ['串本', '白浜'], horizontal=True, label_visibility='collapsed', key='u_loc')

    # ── データ取得 ────────────────────────────────────────────
    with st.spinner('データ取得中...'):
        _u_wx     = _load_weather(_u_loc)
        _u_tide   = _load_tide(_u_loc)
        _u_hourly = _load_hourly()
    _u_ai   = _load_ai_predictions(days=7)
    _u_br   = _load_species_base_rates()
    _u_fc   = _load_forecast_api()
    _u_mw   = _load_morning_wind(_u_loc)
    _u_fw   = _load_forecast_wind(_u_loc)

    # 予測生成日時
    if _PREDICTIONS_PATH.exists():
        import json as _json
        _gen_at = _json.loads(_PREDICTIONS_PATH.read_text(encoding='utf-8')).get('generated_at', '')
        if _gen_at:
            st.caption(f'AI予測更新日時: {_gen_at}')

    if _u_loc == '白浜':
        st.caption('※ 白浜の潮汐は最寄港「田辺」のデータを表示しています。')

    # 風速リスクマップ
    _u_mw_map = {}
    if _u_fw is not None and not _u_fw.empty:
        for _, _r in _u_fw.iterrows(): _u_mw_map[_r['date']] = _r
    if _u_mw is not None and not _u_mw.empty:
        for _, _r in _u_mw.iterrows(): _u_mw_map[_r['date']] = _r

    # AIデータマップ
    _u_ai_map = {p['date']: p for p in _u_ai} if _u_ai else {}

    # 統合日付リスト（最大14日）
    _u_dates = sorted(set(
        (list(_u_wx['date'].unique()) if _u_wx is not None else []) +
        list(_u_ai_map.keys()) +
        (list(_u_hourly['date'].unique()) if _u_hourly is not None else [])
    ))

    def _u_wxe(s):
        if not s: return ''
        if '雪' in s: return '❄️'
        if '雷' in s: return '⛈️'
        if '雨' in s and '晴' in s: return '🌦️'
        if '雨' in s: return '🌧️'
        if '曇' in s and '晴' in s: return '🌤️'
        if '曇' in s: return '☁️'
        if '晴' in s: return '☀️'
        return ''

    def _u_stars(p):
        n = min(3, max(0, round(float(p) * 3)))
        return '★' * n + '☆' * (3 - n)

    if _u_dates:
        st.caption('タップすると詳細を確認できます')
        for _uidx, _ud in enumerate(_u_dates):
            _uwd = _WDAYS[_ud.weekday()]
            _u_ai_p  = _u_ai_map.get(_ud)
            _u_mw_row = _u_mw_map.get(_ud)

            # 天気・気温
            _u_wr = _u_wx[_u_wx['date'] == _ud].iloc[0] if _u_wx is not None and not _u_wx[_u_wx['date'] == _ud].empty else None
            _u_wxe_s = _u_wxe(str(_u_wr['weather']) if _u_wr is not None and pd.notna(_u_wr.get('weather')) else '')
            _u_tmax = int(_u_wr['temp_max']) if _u_wr is not None and pd.notna(_u_wr.get('temp_max')) else None
            _u_tmin = int(_u_wr['temp_min']) if _u_wr is not None and pd.notna(_u_wr.get('temp_min')) else None
            _u_temp_s = f'{_u_tmax}/{_u_tmin}℃' if _u_tmax is not None else ''

            # 潮汐
            _u_tn = _u_tide[(_u_tide['date'] == _ud) & _u_tide['tide_name'].notna()] if _u_tide is not None and not _u_tide.empty else pd.DataFrame()
            _u_tide_s = str(_u_tn.iloc[0]['tide_name']) if not _u_tn.empty else ''

            # 降水
            _u_hd = _u_hourly[_u_hourly['date'] == _ud] if _u_hourly is not None else pd.DataFrame()
            _u_prec = round(float(_u_hd['precipitation_mm'].sum()), 1) if not _u_hd.empty and 'precipitation_mm' in _u_hd.columns else None

            # 出船判断
            if _u_mw_row is not None:
                _u_mw_spd  = _u_mw_row['wind_max_ms']
                _u_mw_prob = _u_mw_row['risk_prob']
                if _u_mw_prob >= 0.90:
                    _u_vs = '✖ STOP'; _u_mw_str = f'休船大 {int(_u_mw_spd)}m/s'
                elif _u_mw_prob >= 0.75:
                    _u_vs = '⚠ CHECK'; _u_mw_str = f'要確認 {int(_u_mw_spd)}m/s'
                else:
                    _u_vs = '✅ GO'; _u_mw_str = f'出船可 {int(_u_mw_spd)}m/s'
            elif _u_ai_p is not None:
                _u_vs = '✅ GO'; _u_mw_spd = None; _u_mw_str = ''
            else:
                _u_vs = None; _u_mw_spd = None; _u_mw_str = ''

            # AI釣果
            _u_gp = _u_ai_p.get('go_proba', 0) if _u_ai_p else None
            _u_ec = _u_ai_p.get('expected_count', 0) if _u_ai_p else None
            _u_star_s = _u_stars(_u_gp) if _u_gp is not None else ''

            # カード色クラス
            _u_drm = 'drm-stop' if _u_vs and '✖' in _u_vs else ('drm-check' if _u_vs and '⚠' in _u_vs else 'drm-go')
            _u_drm_today = 'drm-today' if _ud == _today else ''

            # 潮汐（満潮・干潮時刻）
            _u_td = _u_tide[_u_tide['date'] == _ud] if _u_tide is not None and not _u_tide.empty else pd.DataFrame()
            _u_tide_name = str(_u_td['tide_name'].dropna().iloc[0]) if not _u_td.empty and _u_td['tide_name'].notna().any() else ''
            _u_hi = _u_td[_u_td['type'] == '満潮'].sort_values('time')['time'].tolist() if not _u_td.empty else []
            _u_lo = _u_td[_u_td['type'] == '干潮'].sort_values('time')['time'].tolist() if not _u_td.empty else []
            _u_hi_s = '  '.join(_u_hi[:2]) if _u_hi else ''
            _u_lo_s = '  '.join(_u_lo[:2]) if _u_lo else ''

            # カード上段: 潮型＋気温
            _top_parts = [p for p in [_u_tide_name, _u_temp_s] if p]
            _card_top = '　'.join(_top_parts) if _top_parts else (_u_wxe_s or '--')

            # カード下段1行目: 天気・風
            _r1_parts = [p for p in [_u_wxe_s, (f'🌬 {int(_u_mw_spd)}m/s' if _u_mw_spd is not None else ''), (f'☔ {_u_prec}mm' if _u_prec and _u_prec > 0 else '')] if p]
            _card_r1 = '　'.join(_r1_parts)

            # カード下段2行目: 潮汐時刻
            _r2_parts = [p for p in [(f'満 {_u_hi_s}' if _u_hi_s else ''), (f'干 {_u_lo_s}' if _u_lo_s else '')] if p]
            _card_r2 = '　'.join(_r2_parts)

            # 左カラム
            _star_disp  = _u_star_s if _u_star_s else '---'
            _count_disp = f'{_u_ec:.1f}匹' if _u_ec is not None else '--'
            _today_badge = '<span class="dcard-today-badge">今日</span>' if _ud == _today else ''

            # 出船ストリップ
            if _u_vs:
                _dep_sym = '○' if 'GO' in _u_vs else ('△' if '⚠' in _u_vs else '✕')
                _dep_html = f'<div class="dcard-dep">出船{_dep_sym}</div>'
            else:
                _dep_html = ''

            # HTMLカード組み立て
            _card_html = (
                f'<div class="dcard-wrap {_u_drm} {_u_drm_today}">'
                f'<span class="dcard-marker {_u_drm} {_u_drm_today}"></span>'
                f'<div class="dcard-l">'
                f'<div class="dcard-stars">{_star_disp}</div>'
                f'<div class="dcard-count">{_count_disp}</div>'
                f'<div class="dcard-date">{_ud.month}/{_ud.day}<br><span class="dcard-wd">（{_uwd}）</span></div>'
                f'{_today_badge}'
                f'</div>'
                f'<div class="dcard-r">'
                f'<div class="dcard-top">{_card_top}</div>'
                f'<div class="dcard-bot">'
                f'<div class="dcard-conds">'
                f'<div class="dcard-conds-r1">{_card_r1}</div>'
                f'<div class="dcard-conds-r2">{_card_r2}</div>'
                f'</div>'
                f'{_dep_html}'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(_card_html, unsafe_allow_html=True)
            if st.button('', key=f'unified_{_uidx}', use_container_width=True):
                if _u_ai_p is not None:
                    _show_day_detail(
                        sel_date=_ud, sel_pred=_u_ai_p, mw_row=_u_mw_row,
                        h_wx=_u_wx, h_fc=_u_fc,
                        h_tide=_u_tide, h_br=_u_br,
                        today=_today, wdays=_WDAYS,
                        load_ranking_fn=_load_spot_ranking,
                        build_reasons_fn=_build_reasons,
                        species_lift_fn=_species_lift_str,
                        hourly_df=_u_hourly,
                    )
                else:
                    _show_tab1_detail(
                        sel_date=_ud, wx_df=_u_wx, tide_df=_u_tide,
                        hourly_df=_u_hourly, ai_preds=_u_ai or [],
                        base_rates=_u_br, df_all_data=df_all,
                        location=_u_loc, wdays=_WDAYS,
                        prompt_builder_mod=prompt_builder,
                    )
            # カード間スペーサー（flex gap 16px を1つ稼いで重なりを解消）
            st.markdown('<div class="dcard-spacer"></div>', unsafe_allow_html=True)

        # JS: .dcard-marker の隣ボタンに .dcard-btn クラスを付与
        import streamlit.components.v1 as _stc
        _stc.html("""
        <script>
        (function(){
          function tag(){
            var markers = window.parent.document.querySelectorAll('.dcard-marker');
            markers.forEach(function(m){
              var ec = m.closest('[data-testid="element-container"]');
              if(!ec) return;
              var next = ec.nextElementSibling;
              if(!next) return;
              next.classList.add('dcard-btn-wrap');
              var btn = next.querySelector('button');
              if(btn) btn.classList.add('dcard-btn');
            });
          }
          tag();
          var obs = new MutationObserver(function(){ tag(); });
          obs.observe(window.parent.document.body, {childList:true, subtree:true});
        })();
        </script>
        """, height=0)
    else:
        st.info('データを取得できませんでした。')

    # ── 最近釣れ始めている魚 ─────────────────────────────────────
    st.markdown('---')
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

        _tr_items = list(_trending.itertuples())
        _tr_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:8px;">'
        for _row in _tr_items:
            _r_pct = _row.recent_rate * 100
            _p_pct = _row.prev_rate  * 100
            _ratio = _row.ratio
            _lbl   = _row.label
            _lbl_color = '#c0392b' if '急増' in _lbl else ('#2471a3' if '増加' in _lbl else '#1a7a4a')
            _tr_html += f"""
<div style="background:#FFFFFF;border-radius:12px;padding:16px 18px;
     box-shadow:0 2px 12px rgba(0,0,0,0.07);border-top:4px solid {_lbl_color};">
  <div style="font-size:0.78rem;font-weight:600;color:{_lbl_color};
       letter-spacing:0.05em;margin-bottom:6px;">{_lbl}</div>
  <div style="font-size:1.25rem;font-weight:700;color:#0B3D5C;
       margin-bottom:10px;">{_row.species}</div>
  <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#555;">
    <span>直近</span><span style="font-weight:700;color:{_lbl_color};">{_r_pct:.0f}%</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#888;">
    <span>前期間</span><span>{_p_pct:.0f}%</span>
  </div>
  <div style="margin-top:8px;background:#eee;border-radius:4px;height:5px;overflow:hidden;">
    <div style="background:{_lbl_color};width:{min(_r_pct*2, 100):.0f}%;
         height:5px;border-radius:4px;"></div>
  </div>
  <div style="margin-top:6px;font-size:0.75rem;color:#999;text-align:right;">
    前期間比 {_ratio:.1f}x
  </div>
</div>"""
        _tr_html += '</div>'
        st.markdown(_tr_html, unsafe_allow_html=True)


# ============================================================
# Tab 1: 釣果予測 — 詳細ダイアログ
# ============================================================

@st.dialog('📅 天気・釣果詳細', width='large')
def _show_tab1_detail(sel_date, wx_df, tide_df, hourly_df, ai_preds,
                      base_rates, df_all_data, location, wdays, prompt_builder_mod):
    """天気・潮汐・AI予測をダイアログで表示する。"""
    import streamlit.components.v1 as _stc2
    _stc2.html(
        '<script>(function(){'
        'var SEL="[data-testid=\\"stDialogScrollArea\\"]";'
        'var T0=Date.now();var DUR=5000;'
        'function rst(){'
        '[window.parent,window.top].forEach(function(w){'
        'try{w.document.querySelectorAll(SEL).forEach(function(e){e.scrollTop=0;});}catch(e){}});}'
        '[80,250,600,1200,2500].forEach(function(t){setTimeout(rst,t)});'
        '[window.parent,window.top].forEach(function(w){'
        'try{'
        'var ob=new MutationObserver(function(){'
        'if(Date.now()-T0<DUR){rst();}else{ob.disconnect();}});'
        'ob.observe(w.document.body,{childList:true,subtree:true});'
        '}catch(e){}});'
        '})();</script>',
        height=1, scrolling=False,
    )
    d = sel_date
    wd = wdays[d.weekday()]
    is_today = (d == date.today())
    today_badge = ' <span style="background:#1B8FA8;color:#fff;border-radius:6px;padding:2px 8px;font-size:0.75rem;vertical-align:middle;">今日</span>' if is_today else ''
    st.markdown(f'<h3 style="margin:0 0 14px;">{d.month}/{d.day}（{wd}）{today_badge}</h3>', unsafe_allow_html=True)

    # ── データ準備 ────────────────────────────────────────────
    wx_row = None
    if wx_df is not None:
        wr = wx_df[wx_df['date'] == d]
        wx_row = wr.iloc[0] if not wr.empty else None

    def _wxe2(s):
        if not s: return ''
        s = str(s)
        if '雪' in s: return '❄️'
        if '雷' in s: return '⛈️'
        if '雨' in s and '晴' in s: return '🌦️'
        if '雨' in s: return '🌧️'
        if '曇' in s and '晴' in s: return '🌤️'
        if '曇' in s: return '☁️'
        if '晴' in s: return '☀️'
        return ''

    weather_str = str(wx_row['weather']) if wx_row is not None and pd.notna(wx_row.get('weather')) else '–'
    temp_max = wx_row.get('temp_max') if wx_row is not None else None
    temp_min = wx_row.get('temp_min') if wx_row is not None else None
    sunrise  = str(wx_row['sunrise'])[:5] if wx_row is not None and pd.notna(wx_row.get('sunrise')) else '–'

    hd = hourly_df[hourly_df['date'] == d] if hourly_df is not None else pd.DataFrame()
    prec_total = round(float(hd['precipitation_mm'].sum()), 1) if not hd.empty and 'precipitation_mm' in hd.columns else None

    tide_name_str = '–'
    high_str = low_str = '–'
    if tide_df is not None and not tide_df.empty:
        dt = tide_df[tide_df['date'] == d]
        if not dt.empty:
            tn = dt['tide_name'].dropna()
            if not tn.empty: tide_name_str = str(tn.iloc[0])
            hi2 = dt[dt['type'] == '満潮'].sort_values('time')
            lo2 = dt[dt['type'] == '干潮'].sort_values('time')
            if not hi2.empty: high_str = '　'.join(f"{r['time']}({r['height_cm']}cm)" for _, r in hi2.iterrows())
            if not lo2.empty: low_str  = '　'.join(f"{r['time']}({r['height_cm']}cm)" for _, r in lo2.iterrows())

    ai_pred = next((p for p in ai_preds if p['date'] == d), None) if ai_preds else None

    _HOURS = [0, 6, 12, 18]
    _XRANGE = [-3, 21]
    _XTICKS = dict(tickvals=[0, 6, 12, 18], ticktext=['0:00', '6:00', '12:00', '18:00'])
    _CM = dict(l=45, r=10, t=8, b=8)

    def _hv_d(hour_, col_, fmt='{}'):
        r_ = hd[hd['hour'] == hour_] if not hd.empty else pd.DataFrame()
        if r_.empty or not pd.notna(r_.iloc[0].get(col_)):
            return '–'
        return fmt.format(r_.iloc[0][col_])

    # ── 1. AI釣行予測（7日以内の場合）──────────────────────────
    if ai_pred:
        _gp = ai_pred.get('go_proba', 0)
        _ec = ai_pred.get('expected_count', 0)
        _sp = _species_lift_str(ai_pred.get('species_proba', {}), base_rates, top_n=2)
        if _gp >= 0.6:
            _vc2, _vbg2, _vt = '#1a7a4a', '#d4edda', '✅ GO — 出船できる見込みです'
        elif _gp >= 0.4:
            _vc2, _vbg2, _vt = '#7a5f00', '#fff3cd', '⚠️ CHECK — 条件を確認してください'
        else:
            _vc2, _vbg2, _vt = '#7a1c24', '#f8d7da', '✖ 条件がやや厳しい日です'
        st.markdown(f"""
<div style="background:{_vbg2};border-left:5px solid {_vc2};border-radius:10px;padding:14px 18px;margin-bottom:16px;">
  <div style="font-weight:700;color:{_vc2};margin-bottom:6px;">{_vt}</div>
  <div style="display:flex;flex-wrap:wrap;gap:16px;font-size:0.88rem;color:#333;">
    <span>推奨スコア <b>{_gp*100:.0f}%</b></span>
    <span>期待釣果 <b>{_ec:.1f}匹</b></span>
    {'<span>注目魚種 <b>' + _sp + '</b></span>' if _sp else ''}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 2. 天気・気温・降水・潮 カード ──────────────────────────
    _dc = "background:#fff;border-radius:12px;padding:clamp(10px,3vw,16px) clamp(12px,3.5vw,18px);box-shadow:0 2px 10px rgba(0,0,0,0.08);"
    _dlb = "font-size:clamp(0.6rem,2vw,0.72rem);color:#6B7B8D;font-weight:600;letter-spacing:0.05em;margin-bottom:5px;"
    _dvl = "font-size:clamp(0.95rem,3.5vw,1.25rem);color:#0B3D5C;font-weight:700;line-height:1.2;"
    _tmax_str = f'<span style="color:#e74c3c;font-weight:700;">{int(temp_max)}℃</span>' if temp_max is not None else '–'
    _tmin_str = f'<span style="color:#3498db;font-weight:700;">{int(temp_min)}℃</span>' if temp_min is not None else '–'
    _prec_str = f'{prec_total}mm' if prec_total is not None else '–'
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:clamp(8px,2.5vw,12px);margin-bottom:12px;">
  <div style="{_dc}border-left:4px solid #1B8FA8;"><div style="{_dlb}">{_wxe2(weather_str)} 天気</div><div style="{_dvl}">{weather_str}</div></div>
  <div style="{_dc}border-left:4px solid #e74c3c;"><div style="{_dlb}">🌡️ 気温</div><div style="{_dvl}">{_tmax_str} / {_tmin_str}</div></div>
  <div style="{_dc}border-left:4px solid #3498db;"><div style="{_dlb}">☔ 降水量（日計）</div><div style="{_dvl}">{_prec_str}</div></div>
  <div style="{_dc}border-left:4px solid #22AECB;"><div style="{_dlb}">🌙 潮（{tide_name_str}）</div>
    <div style="font-size:0.8rem;color:#0B3D5C;margin-top:2px;">🔼 {high_str}</div>
    <div style="font-size:0.8rem;color:#0B3D5C;">🔽 {low_str}</div></div>
</div>
<div style="font-size:0.78rem;color:#888;margin-bottom:14px;">🌅 日の出 {sunrise}</div>
""", unsafe_allow_html=True)

    # ── 3. 6時間ごとの詳細 ────────────────────────────────────
    _tc = "padding:4px 8px;text-align:center;font-size:0.76rem;color:#1C3448;"
    _tl = "padding:4px 6px;color:#888;font-weight:600;font-size:0.76rem;white-space:nowrap;"
    _th = "padding:5px 8px;text-align:center;font-size:0.76rem;font-weight:700;color:#0B3D5C;border-bottom:2px solid #DDE8F5;"

    def _tr(label, cells_html):
        return f'<tr><td style="{_tl}">{label}</td>{cells_html}</tr>'

    def _tds(col, fmt='{}'):
        return ''.join(f'<td style="{_tc}">{_hv_d(h, col, fmt)}</td>' for h in _HOURS)

    _wx_tds = ''.join(
        f'<td style="{_tc}">{_wxe2(_hv_d(h,"weather_text")) or _hv_d(h,"weather_text")}</td>'
        for h in _HOURS
    )
    _wind_tds = ''.join(
        f'<td style="{_tc}">{_hv_d(h,"wind_dir")}<br>{_hv_d(h,"wind_speed_ms","{:.0f}m/s")}</td>'
        for h in _HOURS
    )
    _hour_headers = ''.join(f'<th style="{_th}">{h:02d}:00</th>' for h in _HOURS)
    st.markdown(f"""
<div style="overflow-x:auto;margin-bottom:16px;background:#F8FBFF;border-radius:10px;padding:10px 12px;">
<table style="width:100%;min-width:260px;border-collapse:collapse;">
  <thead><tr><th style="{_tl}"></th>{_hour_headers}</tr></thead>
  <tbody>
    {_tr('天気', _wx_tds)}
    {_tr('気温', _tds('temp_c', '{:.0f}℃'))}
    {_tr('湿度', _tds('humidity', '{:.0f}%'))}
    {_tr('風', _wind_tds)}
    {_tr('降水', _tds('precipitation_mm', '{:.1f}mm'))}
  </tbody>
</table>
</div>
""", unsafe_allow_html=True)

    # ── 4. グラフ（ズーム・パン無効）──────────────────────────
    _cfg = {'displayModeBar': False, 'scrollZoom': False, 'staticPlot': False}
    ch_l, ch_r = st.columns(2)
    with ch_l:
        temp_pts = []
        for hour in _HOURS:
            slot = hd[hd['hour'] == hour] if not hd.empty else pd.DataFrame()
            v = slot.iloc[0]['temp_c'] if not slot.empty and pd.notna(slot.iloc[0].get('temp_c')) else None
            temp_pts.append((hour, v))
        if any(v is not None for _, v in temp_pts):
            xs = [h for h, v in temp_pts if v is not None]
            ys = [v for _, v in temp_pts if v is not None]
            fig_t = go.Figure(go.Scatter(x=xs, y=ys, mode='lines+markers',
                line=dict(shape='spline', color='tomato', width=2),
                marker=dict(size=7, color='tomato'), showlegend=False))
            fig_t.update_xaxes(range=_XRANGE, showgrid=True, gridcolor='#eee', zeroline=False,
                fixedrange=True, **_XTICKS)
            fig_t.update_yaxes(title_text='℃', tickformat='.0f', showgrid=True, gridcolor='#eee',
                fixedrange=True)
            fig_t.update_layout(height=180, margin=_CM, plot_bgcolor='white', paper_bgcolor='white',
                title=dict(text='気温', font_size=13, x=0.02), dragmode=False)
            st.plotly_chart(fig_t, use_container_width=True, config=_cfg)

    with ch_r:
        if tide_df is not None and not tide_df.empty:
            dtplot = tide_df[tide_df['date'] == d].copy()
            if not dtplot.empty:
                dtplot['height_cm'] = pd.to_numeric(dtplot['height_cm'], errors='coerce')
                dtplot['hour_f'] = dtplot['time'].apply(
                    lambda t: int(str(t).split(':')[0]) + int(str(t).split(':')[1]) / 60)
                dtplot = dtplot.sort_values('hour_f')
                fig_td = go.Figure(go.Scatter(
                    x=dtplot['hour_f'], y=dtplot['height_cm'], mode='lines+markers',
                    line=dict(shape='spline', color='steelblue', width=2),
                    marker=dict(size=7, color='steelblue'),
                    text=dtplot.apply(lambda r: f"{r['type']} {str(r['time'])[:5]} {r['height_cm']:.0f}cm", axis=1),
                    hovertemplate='%{text}<extra></extra>', showlegend=False))
                fig_td.update_xaxes(range=_XRANGE, showgrid=True, gridcolor='#eee',
                    fixedrange=True, **_XTICKS)
                fig_td.update_yaxes(title_text='cm', tickformat='.0f', showgrid=True, gridcolor='#eee',
                    fixedrange=True)
                fig_td.update_layout(height=180, margin=_CM, plot_bgcolor='white', paper_bgcolor='white',
                    title=dict(text='潮位', font_size=13, x=0.02), dragmode=False)
                st.plotly_chart(fig_td, use_container_width=True, config=_cfg)

    # ── 5. Claudeプロンプト（AI予測ありの場合）─────────────────
    if ai_pred:
        with st.expander('💬 Claude AI へのプロンプトを生成'):
            fc_api = _load_forecast_api()
            fc_row2 = None
            if fc_api is not None:
                fc_m = fc_api[fc_api['date'] == d]
                fc_row2 = fc_m.iloc[0] if not fc_m.empty else None
            wave_fc2 = fc_row2.get('forecast_wave_height_m') if fc_row2 is not None else None
            recent_wt2 = (df_all_data[df_all_data['date'] >= (df_all_data['date'].max() - pd.Timedelta(days=7))]['water_temp_avg'].mean())
            wt2 = float(recent_wt2) if pd.notna(recent_wt2) else None
            _br2 = base_rates
            _sp_raw2 = ai_pred.get('species_proba', {})
            sp_rank2 = sorted([(sp, p / _br2[sp] if _br2.get(sp, 0) > 0 else 1.0) for sp, p in _sp_raw2.items()], key=lambda x: -x[1])[:3]
            pred_weather2 = ai_pred.get('weather')
            if not pred_weather2 and wx_row is not None and pd.notna(wx_row.get('weather')):
                pred_weather2 = str(wx_row['weather'])
            pred_tmax2 = ai_pred.get('temp_max') or (float(wx_row['temp_max']) if wx_row is not None and pd.notna(wx_row.get('temp_max')) else None)
            pred_tmin2 = ai_pred.get('temp_min') or (float(wx_row['temp_min']) if wx_row is not None and pd.notna(wx_row.get('temp_min')) else None)
            sim_mask2 = df_all_data['count'].notna()
            if wave_fc2 is not None and pd.notna(wave_fc2):
                sim_mask2 &= df_all_data['wave_height_m'].between(float(wave_fc2) - 0.5, float(wave_fc2) + 0.5)
            if wt2 is not None:
                sim_mask2 &= df_all_data['water_temp_avg'].between(wt2 - 1, wt2 + 1)
            sim_df2 = df_all_data[sim_mask2]
            prompt_txt = prompt_builder_mod.build_prompt(
                target_date=d, weather=pred_weather2 or '不明',
                temp_max=float(pred_tmax2) if pred_tmax2 is not None else None,
                temp_min=float(pred_tmin2) if pred_tmin2 is not None else None,
                wind_dir=ai_pred.get('wind_dir', '不明'),
                wind_speed_ms=None,
                wave_height_m=(float(wave_fc2) if wave_fc2 is not None and pd.notna(wave_fc2) else None),
                water_temp_c=wt2,
                tide_name=tide_name_str if tide_name_str != '–' else (ai_pred.get('tide_name') or '不明'),
                rising_ratio=ai_pred.get('rising_ratio'),
                precip_1d=ai_pred.get('precip_1d'), precip_2d=ai_pred.get('precip_2d'), precip_3d=ai_pred.get('precip_3d'),
                predicted_count=ai_pred.get('expected_count'),
                go_score_pct=(ai_pred.get('go_proba', 0) * 100 if ai_pred.get('go_proba') is not None else None),
                species_rank=sp_rank2,
                n_similar=len(sim_df2),
                avg_count_similar=(float(sim_df2['count'].mean()) if len(sim_df2) > 0 else None),
            )
            st.text_area('プロンプト（コピーして claude.ai に貼り付け）', value=prompt_txt, height=260, key=f'dlg_prompt_{d}')
            st.caption('上のテキストをコピーして https://claude.ai に貼り付けてください。')


# ============================================================
# Tab 1: 釣果分析
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


with tab1:
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
            )
            fig_rate.update_layout(margin=dict(t=10, b=10), height=280)
            st.plotly_chart(fig_rate, use_container_width=True, config={'displayModeBar': False})

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
            )
            fig_mean.update_layout(margin=dict(t=10, b=10), height=280)
            st.plotly_chart(fig_mean, use_container_width=True, config={'displayModeBar': False})

        st.markdown('---')

        # ── 2. 水温帯別サイズ分布 & 釣れる確率 ──────────────
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
            st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})

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
            st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': False})

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
                    st.plotly_chart(fig_pdp_a, use_container_width=True, config={'displayModeBar': False})

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
                    st.plotly_chart(fig_pdp_b, use_container_width=True, config={'displayModeBar': False})

        # ── 生データ散布図（参考） ────────────────────────────
        with st.expander('生データ散布図（参考）', expanded=False):
            fig_scatter = px.scatter(
                df2, x='water_temp_avg', y='count', color='species_detail',
                hover_data=['date', 'spot', 'angler'],
                labels={'water_temp_avg': '水温平均 (℃)', 'count': '釣果数（匹）',
                        'species_detail': '魚種'},
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

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
                        st.plotly_chart(_fig_sz, use_container_width=True, config={'displayModeBar': False})

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
                            st.plotly_chart(_fig_cnt, use_container_width=True, config={'displayModeBar': False})

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
                            st.plotly_chart(_fig_tsz, use_container_width=True, config={'displayModeBar': False})
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
                            st.plotly_chart(_fig_tcnt, use_container_width=True, config={'displayModeBar': False})
                        else:
                            st.info('匹数データが不足しています。')


# ============================================================
# Tab 2: 釣り場ランキング
# ============================================================
with tab2:
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
        st.plotly_chart(fig_rank, use_container_width=True, config={'displayModeBar': False})

        agg_display = agg.rename(columns={
            'spot': '釣り場', 'total_count': '総釣果数',
            'avg_size': '平均サイズ(cm)', 'max_size': '最大サイズ(cm)',
            'species_list': '魚種',
        })
        agg_display[['平均サイズ(cm)', '最大サイズ(cm)']] = (
            agg_display[['平均サイズ(cm)', '最大サイズ(cm)']].round(1)
        )
        st.dataframe(agg_display, use_container_width=True, hide_index=True)


# ── 分析タブ：月別・季節別（expander）──
with tab1:
    with st.expander('📅 月別・季節別の釣果傾向', expanded=False):
        df4 = df[df['species'].notna() & df['count'].notna()]
        if df4.empty:
            st.info('表示できるデータがありません。')
        else:
            st.caption('各行（魚種）の月別割合。全月合計が100%になるよう正規化。色が濃い月ほどその魚種の釣果が集中している。')
            heat_df = df4.groupby(['month', 'species'])['count'].sum().reset_index()
            heat_pivot = heat_df.pivot(index='species', columns='month', values='count').fillna(0)
            species_total = heat_pivot.sum(axis=1)
            heat_pivot = heat_pivot.div(species_total, axis=0).mul(100).round(1)
            heat_pivot.columns = [f'{m}月' for m in heat_pivot.columns]
            fig_heat = px.imshow(
                heat_pivot,
                labels=dict(x='月', y='魚種', color='割合 (%)'),
                color_continuous_scale='Blues',
                zmin=0, zmax=100,
                aspect='auto',
            )
            _n_sp_heat = len(heat_pivot.index)
            fig_heat.update_layout(margin=dict(t=10, b=10), height=max(320, _n_sp_heat * 36 + 80))
            st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})

            st.markdown('---')
            st.markdown('##### 季節別 魚種構成')
            season_order = ['春', '夏', '秋', '冬']
            available_seasons = [s for s in season_order if s in df4['season'].values]
            selected_season = st.selectbox('季節を選択', available_seasons)
            season_df = df4[df4['season'] == selected_season]
            pie_df = season_df.groupby('species')['count'].sum().reset_index()
            fig_pie = px.pie(pie_df, values='count', names='species', title=f'{selected_season}の魚種構成')
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with st.expander('🌊 波高・天候条件と釣果', expanded=False):
        df5 = df[df['count'].notna()]
        if df5.empty:
            st.info('表示できるデータがありません。')
        else:
            df5_wave = df5[df5['wave_height_m'].notna()].copy()
            df5_wave['wave_band'] = df5_wave['wave_height_m'].apply(lambda w: f'{int(w)}〜{int(w)+1}m')
            if not df5_wave.empty:
                fig_box = px.box(
                    df5_wave.sort_values('wave_height_m'),
                    x='wave_band', y='count',
                    labels={'wave_band': '波高帯', 'count': '釣果数（匹）'},
                    title='波高帯別の釣果数分布',
                )
                st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info('波高データがありません。')

            df5_bubble = df5[
                df5['wave_height_m'].notna() & df5['water_temp_avg'].notna() & df5['weather'].notna()
            ]
            if not df5_bubble.empty:
                fig_bubble = px.scatter(
                    df5_bubble, x='water_temp_avg', y='wave_height_m',
                    size='count', color='weather',
                    hover_data=['date', 'spot', 'species'],
                    labels={'water_temp_avg': '水温平均 (℃)', 'wave_height_m': '波高 (m)', 'weather': '天候', 'count': '釣果数'},
                    title='天候 × 波高 × 水温 バブルチャート',
                )
                st.plotly_chart(fig_bubble, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info('表示に必要なデータが不足しています。')


