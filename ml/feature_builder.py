"""
ml/feature_builder.py — 特徴量エンジニアリング

釣果CSVに気象（雨量・風）・潮汐・時期特徴量を付加して
学習用 DataFrame を生成する。
"""

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------
_LAT = 33.4833
_LON = 135.7833
_TZ  = 'Asia/Tokyo'

_ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
_TIDE_API    = 'https://api.tide736.net/get_tide.php'
_TIDE_PORT   = {'pc': '30', 'hc': '3'}  # 串本

OUTPUT_DIR = Path('output')
ML_DIR     = Path('ml')

# 全モデル共通の特徴量カラム（train.py / predict.py で共有）
BASE_FEATURES = [
    'month', 'weekday', 'season',
    'precip_1d', 'precip_2d', 'precip_3d',
    'wind_ms_max', 'wind_dir_deg',
    'rising_ratio',
    'water_temp_avg', 'water_temp_1d', 'water_temp_2d',
    'wave_height_m',
]
CATEGORICAL_FEATURES = ['spot_enc']


def get_feature_cols(include_species: bool = False) -> list[str]:
    """学習・予測で使う特徴量カラム名リストを返す。"""
    cols = BASE_FEATURES + CATEGORICAL_FEATURES
    if include_species:
        cols = cols + ['species']
    return cols


# ------------------------------------------------------------------
# 公開関数
# ------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """釣果 DataFrame に全特徴量を付加して返す。

    Args:
        df: load_csv() で読み込んだ釣果 DataFrame
            （'date' 列が datetime 型であること）

    Returns:
        特徴量を追加した DataFrame
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 1. 時期特徴量
    df['month']   = df['date'].dt.month
    df['weekday'] = df['date'].dt.dayofweek   # 0=月, 6=日
    df['season']  = df['month'].map(_month_to_season_num)

    unique_dates = sorted(df['date'].dt.date.unique())

    # 2. 気象特徴量（Open-Meteo Archive）
    weather_df = fetch_weather_features(unique_dates)
    if weather_df is not None:
        df = _left_merge_by_date(df, weather_df)

    # 3. 潮汐特徴量（tide736.net）
    tide_df = fetch_tide_features(unique_dates)
    if tide_df is not None:
        df = _left_merge_by_date(df, tide_df)

    # 4. water_temp_avg の欠損を 7 日移動平均で補完
    df = _fill_water_temp(df)

    # 5. 水温ラグ特徴量（補完済み水温から前日・前々日を計算）
    df = _add_water_temp_lags(df)

    # 6. spot エンコード（10件未満 → 「その他」）
    df = _encode_spot(df)

    return df


def load_csv() -> pd.DataFrame:
    """output/ 以下の全 CSV を連結・重複排除して返す。"""
    csvs = sorted(OUTPUT_DIR.glob('*_rockshore.csv'))
    if not csvs:
        raise FileNotFoundError('output/ に CSV が見つかりません')
    dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset=['date', 'spot', 'angler', 'species'], keep='last', inplace=True)
    df['date']           = pd.to_datetime(df['date'], errors='coerce')
    df['count']          = pd.to_numeric(df['count'], errors='coerce')
    df['water_temp_avg'] = pd.to_numeric(df['water_temp_avg'], errors='coerce')
    df['wave_height_m']  = pd.to_numeric(df['wave_height_m'], errors='coerce')
    df['size_min_cm']    = pd.to_numeric(df['size_min_cm'], errors='coerce')
    df['size_max_cm']    = pd.to_numeric(df['size_max_cm'], errors='coerce')
    return df


# ------------------------------------------------------------------
# 気象特徴量
# ------------------------------------------------------------------

def fetch_weather_features(dates: list[date]) -> pd.DataFrame | None:
    """Open-Meteo Archive から雨量・風速・風向を一括取得。

    Args:
        dates: 釣行日リスト

    Returns:
        columns: date, precip_1d, precip_2d, precip_3d,
                 wind_ms_max, wind_dir_deg
        失敗時は None
    """
    if not dates:
        return None

    start = min(dates) - timedelta(days=3)
    end   = max(dates)

    try:
        resp = requests.get(_ARCHIVE_URL, params={
            'latitude':   _LAT,
            'longitude':  _LON,
            'timezone':   _TZ,
            'start_date': start.isoformat(),
            'end_date':   end.isoformat(),
            'daily': ','.join([
                'precipitation_sum',
                'windspeed_10m_max',
                'winddirection_10m_dominant',
            ]),
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning('Archive API 失敗: %s', exc)
        return None

    d = data['daily']
    arch = pd.DataFrame({
        'date':         [pd.to_datetime(t).date() for t in d['time']],
        'precip':       d.get('precipitation_sum'),
        'wind_ms_max':  d.get('windspeed_10m_max'),
        'wind_dir_deg': d.get('winddirection_10m_dominant'),
    }).set_index('date')

    rows = []
    for fishing_date in dates:
        row: dict = {'date': fishing_date}
        for lag in (1, 2, 3):
            d_lag = fishing_date - timedelta(days=lag)
            row[f'precip_{lag}d'] = (
                float(arch.loc[d_lag, 'precip'])
                if d_lag in arch.index and pd.notna(arch.loc[d_lag, 'precip'])
                else None
            )
        for col in ('wind_ms_max', 'wind_dir_deg'):
            row[col] = (
                float(arch.loc[fishing_date, col])
                if fishing_date in arch.index and pd.notna(arch.loc[fishing_date, col])
                else None
            )
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 潮汐特徴量
# ------------------------------------------------------------------

def fetch_tide_features(dates: list[date]) -> pd.DataFrame | None:
    """tide736.net から潮名・上り潮割合を取得。

    Args:
        dates: 釣行日リスト

    Returns:
        columns: date, tide_name, rising_ratio
        失敗が多い場合は None
    """
    if not dates:
        return None

    rows = []
    for i, fishing_date in enumerate(dates):
        try:
            resp = requests.get(_TIDE_API, params={
                **_TIDE_PORT,
                'yr': str(fishing_date.year),
                'mn': str(fishing_date.month),
                'dy': str(fishing_date.day),
                'rg': 'day',
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            chart = data.get('tide', {}).get('chart', {})
            day_data = chart.get(fishing_date.isoformat(), {})

            tide_name = day_data.get('moon', {}).get('title', '')

            # 全イベント（満潮 + 干潮）
            events: list[dict] = []
            for entry in day_data.get('flood', []):
                events.append({'time': entry['time'], 'type': 'flood',
                                'height_cm': float(entry['cm'])})
            for entry in day_data.get('edd', []):
                events.append({'time': entry['time'], 'type': 'ebb',
                                'height_cm': float(entry['cm'])})

            rising_ratio = calc_rising_ratio(events)

            rows.append({
                'date':         fishing_date,
                'tide_name':    tide_name,
                'rising_ratio': rising_ratio,
            })

        except Exception as exc:
            logger.warning('潮汐取得失敗 %s: %s', fishing_date, exc)
            rows.append({'date': fishing_date, 'tide_name': None, 'rising_ratio': None})

        if i < len(dates) - 1:
            time.sleep(0.3)

    return pd.DataFrame(rows) if rows else None


def calc_rising_ratio(events: list[dict],
                      start_h: int = 7,
                      end_h: int = 14) -> float:
    """満潮・干潮イベントリストから指定時間帯の上り潮割合を算出。

    Args:
        events:  [{'time': 'HH:MM', 'type': 'flood'/'ebb', 'height_cm': float}]
        start_h: 開始時（デフォルト 7）
        end_h:   終了時（デフォルト 14）

    Returns:
        0.0〜1.0。上り潮時間 ÷ 総時間。データなし時は 0.5。
    """
    if not events:
        return 0.5

    def to_min(t: str) -> float:
        h, m = map(int, t.split(':'))
        return h * 60.0 + m

    sorted_ev = sorted(events, key=lambda e: to_min(e['time']))
    start_m = float(start_h * 60)
    end_m   = float(end_h   * 60)

    # 0:00 と 23:59 に境界を補完（交互に flood/ebb が来る前提）
    first_type = 'ebb'   if sorted_ev[0]['type'] == 'flood' else 'flood'
    last_type  = 'flood' if sorted_ev[-1]['type'] == 'ebb'  else 'ebb'
    extended = (
        [{'time': '00:00', 'type': first_type}]
        + sorted_ev
        + [{'time': '23:59', 'type': last_type}]
    )

    rising_min = 0.0
    for i in range(len(extended) - 1):
        a, b = extended[i], extended[i + 1]
        if not (a['type'] == 'ebb' and b['type'] == 'flood'):
            continue  # 下り or 同種が続く区間はスキップ
        a_m = to_min(a['time'])
        b_m = to_min(b['time'])
        overlap_s = max(a_m, start_m)
        overlap_e = min(b_m, end_m)
        if overlap_e > overlap_s:
            rising_min += overlap_e - overlap_s

    return rising_min / (end_m - start_m)


# ------------------------------------------------------------------
# 内部ユーティリティ
# ------------------------------------------------------------------

def _month_to_season_num(m: int) -> int:
    """月 → 季節番号（0=春, 1=夏, 2=秋, 3=冬）"""
    if m in (3, 4, 5):   return 0
    if m in (6, 7, 8):   return 1
    if m in (9, 10, 11): return 2
    return 3


def _left_merge_by_date(df: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """df['date']（datetime）と right['date']（date）でマージ。"""
    right = right.copy()
    right['_date_key'] = pd.to_datetime(right['date'])
    df = df.merge(
        right.drop(columns='date'),
        left_on='date', right_on='_date_key',
        how='left',
    ).drop(columns='_date_key')
    return df


def _fill_water_temp(df: pd.DataFrame) -> pd.DataFrame:
    """water_temp_avg 欠損を 7 日移動平均で補完（日別平均→前方補完）。"""
    df = df.copy()
    if 'water_temp_avg' not in df.columns:
        return df

    # 日別平均を計算してローリング
    daily_avg = (
        df.groupby(df['date'].dt.date)['water_temp_avg']
        .mean()
        .reset_index()
        .rename(columns={'date': '_d', 'water_temp_avg': '_wt_fill'})
    )
    daily_avg['_d'] = pd.to_datetime(daily_avg['_d'])
    daily_avg = daily_avg.set_index('_d').sort_index()
    daily_avg['_wt_rolling'] = (
        daily_avg['_wt_fill']
        .rolling(window=7, min_periods=1)
        .mean()
    )

    # マージして欠損補完
    df = df.merge(
        daily_avg[['_wt_rolling']].reset_index().rename(columns={'_d': '_dt'}),
        left_on='date', right_on='_dt', how='left',
    ).drop(columns='_dt')
    df['water_temp_avg'] = df['water_temp_avg'].fillna(df['_wt_rolling'])
    df = df.drop(columns='_wt_rolling')
    return df


def _add_water_temp_lags(df: pd.DataFrame) -> pd.DataFrame:
    """水温の前日・前々日ラグ特徴量を付加する（補完済み水温から計算）。"""
    df = df.copy()
    daily_wt = (
        df.groupby(df['date'].dt.date)['water_temp_avg']
        .mean()
        .reset_index()
        .rename(columns={'date': '_d', 'water_temp_avg': '_wt'})
        .sort_values('_d')
    )
    daily_wt['water_temp_1d'] = daily_wt['_wt'].shift(1)
    daily_wt['water_temp_2d'] = daily_wt['_wt'].shift(2)
    daily_wt['_d'] = pd.to_datetime(daily_wt['_d'])
    df = df.merge(
        daily_wt[['_d', 'water_temp_1d', 'water_temp_2d']],
        left_on='date', right_on='_d', how='left',
    ).drop(columns='_d')
    return df


def _encode_spot(df: pd.DataFrame) -> pd.DataFrame:
    """
    spot 列をエンコード。
    10件以上の磯は個別カテゴリ、それ未満は「その他」に集約。
    エンコードマップを ml/models/spot_map.json に保存する。
    """
    df = df.copy()
    spot_counts = df['spot'].value_counts()
    major_spots = set(spot_counts[spot_counts >= 10].index)

    df['spot_enc'] = df['spot'].apply(
        lambda s: s if pd.notna(s) and s in major_spots else 'その他'
    )

    # カテゴリ型（LightGBM はそのまま扱える）
    df['spot_enc'] = df['spot_enc'].astype('category')
    df['species']  = df['species'].astype('category')

    # エンコードマップを保存（predict 時に使用）
    models_dir = ML_DIR / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    spot_map = {'major_spots': sorted(major_spots)}
    (models_dir / 'spot_map.json').write_text(
        json.dumps(spot_map, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    logger.info('spot_map.json 保存: %d 磯（個別）+ その他', len(major_spots))

    return df


# ------------------------------------------------------------------
# CLI（単体実行でCSVを出力）
# ------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    logger.info('CSVを読み込み中...')
    raw_df = load_csv()
    logger.info('レコード数: %d', len(raw_df))

    logger.info('特徴量を構築中...')
    feat_df = build_features(raw_df)

    out_path = ML_DIR / 'features.parquet'
    feat_df.to_parquet(out_path, index=False)
    logger.info('特徴量保存完了: %s', out_path)

    # サマリー表示
    print('\n=== 特徴量サマリー ===')
    print(feat_df[['date', 'month', 'weekday', 'season',
                   'precip_1d', 'precip_2d', 'precip_3d',
                   'wind_ms_max', 'wind_dir_deg',
                   'tide_name', 'rising_ratio',
                   'water_temp_avg', 'spot_enc', 'species', 'count']
                  ].describe(include='all').T.to_string())
