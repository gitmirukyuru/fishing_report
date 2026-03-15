"""
ml/predict.py — 予測日の特徴量生成 → モデル推論

dashboard.py から呼び出す予測 API。
"""

import json
import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ml.feature_builder import (
    calc_rising_ratio,
    fetch_tide_features,
    get_feature_cols,
    _month_to_season_num,
)

logger = logging.getLogger(__name__)

MODELS_DIR = Path('ml/models')
_LAT = 33.4833
_LON = 135.7833
_TZ  = 'Asia/Tokyo'
_FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
_MARINE_URL   = 'https://marine-api.open-meteo.com/v1/marine'
_ARCHIVE_URL  = 'https://archive-api.open-meteo.com/v1/archive'


# ------------------------------------------------------------------
# モデルのロード（キャッシュ）
# ------------------------------------------------------------------

_cache: dict = {}
_wt_cache: dict[date, float | None] = {}


def _get_recent_water_temp(d: date) -> float | None:
    """指定日の水温平均を釣果CSVから取得する（キャッシュ付き）。"""
    if d in _wt_cache:
        return _wt_cache[d]
    try:
        csvs = sorted(Path('output').glob('*_rockshore.csv'))
        if not csvs:
            _wt_cache[d] = None
            return None
        dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in csvs]
        df = pd.concat(dfs, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['water_temp_avg'] = pd.to_numeric(df['water_temp_avg'], errors='coerce')
        vals = df.loc[df['date'].dt.date == d, 'water_temp_avg'].dropna()
        result = float(vals.mean()) if not vals.empty else None
    except Exception:
        result = None
    _wt_cache[d] = result
    return result


def _load_models() -> tuple:
    """model_a, model_b, model_c, spot_map を返す（キャッシュ付き）。"""
    if _cache:
        return _cache['a'], _cache['b'], _cache['c'], _cache['spot_map']

    def _load(name: str):
        p = MODELS_DIR / name
        if not p.exists():
            return None
        with open(p, 'rb') as f:
            return pickle.load(f)

    _cache['a'] = _load('model_a.pkl')
    _cache['b'] = _load('model_b.pkl')
    _cache['c'] = _load('model_c.pkl') or {}
    spot_map_path = MODELS_DIR / 'spot_map.json'
    _cache['spot_map'] = (
        json.loads(spot_map_path.read_text(encoding='utf-8'))
        if spot_map_path.exists() else {'major_spots': []}
    )
    return _cache['a'], _cache['b'], _cache['c'], _cache['spot_map']


def models_exist() -> bool:
    """学習済みモデルが存在するか確認。"""
    return (MODELS_DIR / 'model_a.pkl').exists()


# ------------------------------------------------------------------
# 予測日の特徴量ベクトル構築
# ------------------------------------------------------------------

def build_predict_features(
    target_date: date,
    spot: str | None = None,
    species: str | None = None,
    water_temp_avg: float | None = None,
    wave_height_m: float | None = None,
) -> pd.DataFrame | None:
    """予測対象日の特徴量 DataFrame（1行）を構築する。

    Args:
        target_date:    予測日
        spot:           釣り場名（None = 全磯平均を想定）
        species:        魚種（Model A で使用）
        water_temp_avg: 水温（指定なければ 7 日移動平均で代替）
        wave_height_m:  波高（指定なければ Open-Meteo marine 予報値）

    Returns:
        特徴量 DataFrame（1行）。取得失敗時は None。
    """
    row: dict = {}

    # 時期特徴量
    row['month']   = target_date.month
    row['weekday'] = target_date.weekday()
    row['season']  = _month_to_season_num(target_date.month)

    # 気象特徴量（予報 + 直近アーカイブ）
    wx = _fetch_forecast_for_date(target_date)
    if wx is None:
        logger.warning('気象予報取得失敗: %s', target_date)
        wx = {}

    row['wind_ms_max']   = wx.get('wind_ms_max')
    row['wind_dir_deg']  = wx.get('wind_dir_deg')
    row['precip_1d']     = wx.get('precip_1d')
    row['precip_2d']     = wx.get('precip_2d')
    row['precip_3d']     = wx.get('precip_3d')

    # 波高（引数 > 予報値）
    row['wave_height_m'] = wave_height_m if wave_height_m is not None else wx.get('wave_height_m')

    # 水温（引数 > 移動平均）
    row['water_temp_avg'] = water_temp_avg

    # 水温ラグ（過去CSVから前日・前々日の実績水温を取得）
    row['water_temp_1d'] = _get_recent_water_temp(target_date - timedelta(days=1))
    row['water_temp_2d'] = _get_recent_water_temp(target_date - timedelta(days=2))

    # 潮汐特徴量
    tide_df = fetch_tide_features([target_date])
    if tide_df is not None and not tide_df.empty:
        row['tide_name']    = tide_df.iloc[0].get('tide_name')
        row['rising_ratio'] = tide_df.iloc[0].get('rising_ratio')
    else:
        row['tide_name']    = None
        row['rising_ratio'] = 0.5

    # spot エンコード
    _, _, _, spot_map = _load_models()
    major_spots = spot_map.get('major_spots', [])
    row['spot_enc'] = spot if (spot and spot in major_spots) else 'その他'

    # species
    row['species'] = species

    df = pd.DataFrame([row])

    # 数値列を確実に float 型に変換（None は NaN になる）
    num_cols = [
        'month', 'weekday', 'season',
        'precip_1d', 'precip_2d', 'precip_3d',
        'wind_ms_max', 'wind_dir_deg',
        'rising_ratio', 'water_temp_avg', 'water_temp_1d', 'water_temp_2d',
        'wave_height_m',
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    df['spot_enc'] = df['spot_enc'].astype('category')
    if 'species' in df.columns:
        df['species'] = df['species'].astype('category')
    return df


# ------------------------------------------------------------------
# 予測
# ------------------------------------------------------------------

def predict_day(
    target_date: date,
    spot: str | None = None,
    water_temp_avg: float | None = None,
    wave_height_m: float | None = None,
) -> dict | None:
    """指定日の釣行予測を返す。

    Returns:
        {
          'date':             date,
          'expected_count':   float,     # 期待釣果数（匹）
          'go_proba':         float,     # 釣行推奨スコア 0〜1
          'species_proba':    {species: float},  # 魚種別出現確率（上位3種）
        }
        モデル未学習時は None。
    """
    if not models_exist():
        return None

    model_a, model_b, model_c, _ = _load_models()
    feat_cols_a = get_feature_cols(include_species=True)
    feat_cols_b = get_feature_cols(include_species=False)

    results: dict = {'date': target_date}

    # --- Model A: 釣果数回帰 ---
    if model_a is not None:
        feat_a = build_predict_features(target_date, spot=spot,
                                        water_temp_avg=water_temp_avg,
                                        wave_height_m=wave_height_m)
        if feat_a is not None:
            X_a = feat_a[[c for c in feat_cols_a if c in feat_a.columns]]
            _align_categories(X_a, model_a)
            pred_count = float(model_a.predict(X_a)[0])
            results['expected_count'] = max(0.0, round(pred_count, 1))

    # --- Model B: 釣行推奨 ---
    if model_b is not None:
        feat_b = build_predict_features(target_date, spot=spot,
                                        water_temp_avg=water_temp_avg,
                                        wave_height_m=wave_height_m)
        if feat_b is not None:
            X_b = feat_b[[c for c in feat_cols_b if c in feat_b.columns]]
            _align_categories(X_b, model_b)
            proba = float(model_b.predict_proba(X_b)[0][1])
            results['go_proba'] = round(proba, 3)

    # --- Model C: 魚種別出現確率 ---
    if model_c:
        species_proba: dict[str, float] = {}
        feat_c = build_predict_features(target_date, spot=spot,
                                        water_temp_avg=water_temp_avg,
                                        wave_height_m=wave_height_m)
        if feat_c is not None:
            X_c = feat_c[[c for c in feat_cols_b if c in feat_c.columns]]
            for sp, m in model_c.items():
                _align_categories(X_c, m)
                try:
                    p = float(m.predict_proba(X_c)[0][1])
                    species_proba[sp] = round(p, 3)
                except Exception:
                    pass

        # 確率上位3種
        top3 = sorted(species_proba.items(), key=lambda x: -x[1])[:3]
        results['species_proba'] = dict(top3)

    return results


def predict_multi_days(
    days: int = 7,
    spot: str | None = None,
    water_temp_avg: float | None = None,
) -> list[dict]:
    """向こう N 日間の予測を返す（API 呼び出しをバッチ化）。"""
    today = date.today()
    target_dates = [today + timedelta(days=i) for i in range(days)]

    # 事前にバッチで気象・潮汐を取得
    batch_wx   = _fetch_batch_weather(target_dates)
    batch_tide = _fetch_batch_tide(target_dates)

    results = []
    for d in target_dates:
        wx   = batch_wx.get(d, {})
        tide = batch_tide.get(d, {})
        r = _predict_day_from_batch(d, wx, tide, spot=spot,
                                    water_temp_avg=water_temp_avg)
        if r:
            results.append(r)
    return results


def predict_spot_ranking(
    target_date: date,
    water_temp_avg: float | None = None,
) -> pd.DataFrame:
    """全磯の期待釣果スコアをバッチ予測してランキング DataFrame を返す。

    気象・潮汐は1回だけ取得し、全磯への推論を一括実行する。

    Returns:
        columns: spot, expected_count, go_proba
    """
    if not models_exist():
        return pd.DataFrame()

    model_a, model_b, _, spot_map = _load_models()
    major_spots = [s for s in spot_map.get('major_spots', []) if s != 'その他']

    if not major_spots or (model_a is None and model_b is None):
        return pd.DataFrame()

    # 気象・潮汐を1回だけ取得
    wx   = _fetch_batch_weather([target_date]).get(target_date, {})
    tide = _fetch_batch_tide([target_date]).get(target_date, {})

    # 全磯の特徴量を一括構築
    num_cols = [
        'month', 'weekday', 'season',
        'precip_1d', 'precip_2d', 'precip_3d',
        'wind_ms_max', 'wind_dir_deg',
        'rising_ratio', 'water_temp_avg', 'water_temp_1d', 'water_temp_2d',
        'wave_height_m',
    ]
    base_row = {
        'month':         target_date.month,
        'weekday':       target_date.weekday(),
        'season':        _month_to_season_num(target_date.month),
        'precip_1d':     wx.get('precip_1d'),
        'precip_2d':     wx.get('precip_2d'),
        'precip_3d':     wx.get('precip_3d'),
        'wind_ms_max':   wx.get('wind_ms_max'),
        'wind_dir_deg':  wx.get('wind_dir_deg'),
        'wave_height_m': wx.get('wave_height_m'),
        'water_temp_avg': water_temp_avg,
        'water_temp_1d': _get_recent_water_temp(target_date - timedelta(days=1)),
        'water_temp_2d': _get_recent_water_temp(target_date - timedelta(days=2)),
        'rising_ratio':  tide.get('rising_ratio', 0.5),
        'species':       None,
    }

    feat_rows = [{**base_row, 'spot_enc': sp} for sp in major_spots]
    feat_df = pd.DataFrame(feat_rows)

    for col in num_cols:
        feat_df[col] = pd.to_numeric(feat_df[col], errors='coerce').astype(float)
    feat_df['spot_enc'] = feat_df['spot_enc'].astype('category')
    feat_df['species']  = feat_df['species'].astype('category')

    feat_cols_a = get_feature_cols(include_species=True)
    feat_cols_b = get_feature_cols(include_species=False)

    result_df = pd.DataFrame({'spot': major_spots})

    if model_a is not None:
        X_a = feat_df[[c for c in feat_cols_a if c in feat_df.columns]].copy()
        preds = model_a.predict(X_a)
        result_df['expected_count'] = [max(0.0, round(float(p), 1)) for p in preds]
    else:
        result_df['expected_count'] = 0.0

    if model_b is not None:
        X_b = feat_df[[c for c in feat_cols_b if c in feat_df.columns]].copy()
        probas = model_b.predict_proba(X_b)[:, 1]
        result_df['go_proba'] = [round(float(p), 3) for p in probas]
    else:
        result_df['go_proba'] = 0.0

    return result_df.sort_values('expected_count', ascending=False).reset_index(drop=True)


def compute_water_temp_pdp(
    temp_step: float = 0.5,
) -> pd.DataFrame | None:
    """水温の Partial Dependence Plot データを計算する。

    学習データの全レコードに対して、水温だけを変化させたときの
    モデル予測平均値を計算する（他の特徴量は実データのまま）。

    Returns:
        columns: water_temp, expected_count, go_proba
        モデル未学習またはfeatures.parquet未生成時は None
    """
    if not models_exist():
        return None

    feat_path = Path('ml/features.parquet')
    if not feat_path.exists():
        return None

    model_a, model_b, _, _ = _load_models()
    feat_df = pd.read_parquet(feat_path)

    feat_cols_a = get_feature_cols(include_species=True)
    feat_cols_b = get_feature_cols(include_species=False)

    # カテゴリ列を復元
    for col in ['spot_enc', 'species']:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].astype('category')

    X_base_a = feat_df[[c for c in feat_cols_a if c in feat_df.columns]].copy()
    X_base_b = feat_df[[c for c in feat_cols_b if c in feat_df.columns]].copy()

    t_min = float(np.floor(feat_df['water_temp_avg'].quantile(0.05)))
    t_max = float(np.ceil(feat_df['water_temp_avg'].quantile(0.95)))
    temps = np.arange(t_min, t_max + temp_step, temp_step)

    rows = []
    for t in temps:
        row: dict = {'water_temp': round(float(t), 2)}

        if model_a is not None and 'water_temp_avg' in X_base_a.columns:
            X_a = X_base_a.copy()
            X_a['water_temp_avg'] = t
            row['expected_count'] = round(float(np.maximum(0, model_a.predict(X_a)).mean()), 2)

        if model_b is not None and 'water_temp_avg' in X_base_b.columns:
            X_b = X_base_b.copy()
            X_b['water_temp_avg'] = t
            row['go_proba'] = round(float(model_b.predict_proba(X_b)[:, 1].mean()), 3)

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 気象予報取得（予測日用）
# ------------------------------------------------------------------

def _fetch_batch_weather(dates: list[date]) -> dict[date, dict]:
    """複数日の気象を一括取得して {date: {key: value}} を返す。

    - 風速・風向・波高: forecast API（16日分）
    - D-1〜D-3 雨量: forecast API past_days + 一括 archive
    """
    today = date.today()
    result: dict[date, dict] = {d: {} for d in dates}

    max_days = (max(dates) - today).days + 1
    past_days = 3  # D-1〜D-3 をカバー

    # 風速・風向（forecast daily）
    try:
        resp = requests.get(_FORECAST_URL, params={
            'latitude':   _LAT, 'longitude': _LON, 'timezone': _TZ,
            'daily': 'windspeed_10m_max,winddirection_10m_dominant',
            'forecast_days': max(max_days, 1),
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()['daily']
        for t, ws, wd in zip(
            data.get('time', []),
            data.get('windspeed_10m_max', []),
            data.get('winddirection_10m_dominant', []),
        ):
            d = pd.to_datetime(t).date()
            if d in result:
                result[d]['wind_ms_max']  = float(ws) if ws is not None else None
                result[d]['wind_dir_deg'] = float(wd) if wd is not None else None
    except Exception as exc:
        logger.warning('Forecast 風速 API 失敗: %s', exc)

    # 波高（marine daily）
    try:
        resp = requests.get(_MARINE_URL, params={
            'latitude':   _LAT, 'longitude': _LON, 'timezone': _TZ,
            'daily': 'wave_height_max',
            'forecast_days': max(max_days, 1),
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()['daily']
        for t, wh in zip(data.get('time', []), data.get('wave_height_max', [])):
            d = pd.to_datetime(t).date()
            if d in result:
                result[d]['wave_height_m'] = float(wh) if wh is not None else None
    except Exception as exc:
        logger.warning('Marine API 失敗: %s', exc)

    # 雨量: forecast API の past_days を使って直近3日を取得
    try:
        resp = requests.get(_FORECAST_URL, params={
            'latitude':    _LAT, 'longitude': _LON, 'timezone': _TZ,
            'daily':       'precipitation_sum',
            'forecast_days': max(max_days, 1),
            'past_days':   past_days,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()['daily']
        precip_map: dict[date, float | None] = {}
        for t, p in zip(data.get('time', []), data.get('precipitation_sum', [])):
            precip_map[pd.to_datetime(t).date()] = float(p) if p is not None else None

        for d in dates:
            for lag in (1, 2, 3):
                key = f'precip_{lag}d'
                lag_d = d - timedelta(days=lag)
                result[d][key] = precip_map.get(lag_d)
    except Exception as exc:
        logger.warning('Forecast 雨量 API 失敗: %s', exc)

    return result


def _fetch_batch_tide(dates: list[date]) -> dict[date, dict]:
    """複数日の潮汐特徴量を一括取得して {date: {key: value}} を返す。"""
    if not dates:
        return {}

    from ml.feature_builder import fetch_tide_features
    tide_df = fetch_tide_features(dates)
    result: dict[date, dict] = {}

    if tide_df is not None and not tide_df.empty:
        for _, row in tide_df.iterrows():
            d = row['date'] if isinstance(row['date'], date) else row['date'].date()
            result[d] = {
                'tide_name':    row.get('tide_name'),
                'rising_ratio': row.get('rising_ratio'),
            }
    return result


def _predict_day_from_batch(
    target_date: date,
    wx: dict,
    tide: dict,
    spot: str | None = None,
    water_temp_avg: float | None = None,
) -> dict | None:
    """バッチ取得済みの気象・潮汐から予測を実行する。"""
    if not models_exist():
        return None

    model_a, model_b, model_c, spot_map = _load_models()
    feat_cols_a = get_feature_cols(include_species=True)
    feat_cols_b = get_feature_cols(include_species=False)

    # 特徴量ベクトルを構築
    major_spots = spot_map.get('major_spots', [])
    spot_enc = spot if (spot and spot in major_spots) else 'その他'

    row: dict = {
        'month':        target_date.month,
        'weekday':      target_date.weekday(),
        'season':       _month_to_season_num(target_date.month),
        'precip_1d':    wx.get('precip_1d'),
        'precip_2d':    wx.get('precip_2d'),
        'precip_3d':    wx.get('precip_3d'),
        'wind_ms_max':  wx.get('wind_ms_max'),
        'wind_dir_deg': wx.get('wind_dir_deg'),
        'wave_height_m': wx.get('wave_height_m'),
        'water_temp_avg': water_temp_avg,
        'water_temp_1d': _get_recent_water_temp(target_date - timedelta(days=1)),
        'water_temp_2d': _get_recent_water_temp(target_date - timedelta(days=2)),
        'rising_ratio': tide.get('rising_ratio', 0.5),
        'tide_name':    tide.get('tide_name'),
        'spot_enc':     spot_enc,
        'species':      None,
    }

    feat = pd.DataFrame([row])
    num_cols = [
        'month', 'weekday', 'season',
        'precip_1d', 'precip_2d', 'precip_3d',
        'wind_ms_max', 'wind_dir_deg',
        'rising_ratio', 'water_temp_avg', 'water_temp_1d', 'water_temp_2d',
        'wave_height_m',
    ]
    for col in num_cols:
        feat[col] = pd.to_numeric(feat[col], errors='coerce').astype(float)
    feat['spot_enc'] = feat['spot_enc'].astype('category')
    feat['species']  = feat['species'].astype('category')

    results: dict = {
        'date':         target_date,
        'tide_name':    row['tide_name'],
        'rising_ratio': row['rising_ratio'],
        'precip_1d':    row['precip_1d'],
        'precip_2d':    row['precip_2d'],
        'precip_3d':    row['precip_3d'],
        'wind_ms_max':  row.get('wind_ms_max'),
        'wind_dir_deg': row.get('wind_dir_deg'),
    }

    # Model A
    if model_a is not None:
        X_a = feat[[c for c in feat_cols_a if c in feat.columns]].copy()
        _align_categories(X_a, model_a)
        pred = float(model_a.predict(X_a)[0])
        results['expected_count'] = max(0.0, round(pred, 1))

    # Model B
    if model_b is not None:
        X_b = feat[[c for c in feat_cols_b if c in feat.columns]].copy()
        _align_categories(X_b, model_b)
        results['go_proba'] = round(float(model_b.predict_proba(X_b)[0][1]), 3)

    # Model C
    if model_c:
        species_proba: dict[str, float] = {}
        X_c = feat[[c for c in feat_cols_b if c in feat.columns]].copy()
        for sp, m in model_c.items():
            _align_categories(X_c, m)
            try:
                p = float(m.predict_proba(X_c)[0][1])
                species_proba[sp] = round(p, 3)
            except Exception:
                pass
        top3 = sorted(species_proba.items(), key=lambda x: -x[1])[:3]
        results['species_proba'] = dict(top3)

    return results


def _fetch_forecast_for_date(target_date: date) -> dict | None:
    """指定日の気象予報・過去雨量を取得する。"""
    today = date.today()
    result: dict = {}

    # 未来日付: Open-Meteo Forecast API
    if target_date >= today:
        _fetch_future_weather(target_date, result)
    else:
        _fetch_past_weather(target_date, result)

    # D-1〜D-3 雨量は Archive から
    _fetch_precip_history(target_date, result)

    return result


def _fetch_future_weather(target_date: date, result: dict) -> None:
    """Open-Meteo forecast + marine API から予報値を取得。"""
    today = date.today()
    forecast_days = (target_date - today).days + 1

    if forecast_days < 1 or forecast_days > 16:
        return

    try:
        resp = requests.get(_FORECAST_URL, params={
            'latitude':   _LAT, 'longitude': _LON, 'timezone': _TZ,
            'daily': 'windspeed_10m_max,winddirection_10m_dominant',
            'forecast_days': forecast_days,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()['daily']
        idx = forecast_days - 1
        result['wind_ms_max']  = _safe_get(data.get('windspeed_10m_max', []), idx)
        result['wind_dir_deg'] = _safe_get(data.get('winddirection_10m_dominant', []), idx)
    except Exception as exc:
        logger.warning('Forecast API 失敗: %s', exc)

    try:
        resp = requests.get(_MARINE_URL, params={
            'latitude':   _LAT, 'longitude': _LON, 'timezone': _TZ,
            'daily': 'wave_height_max',
            'forecast_days': forecast_days,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()['daily']
        idx = forecast_days - 1
        result['wave_height_m'] = _safe_get(data.get('wave_height_max', []), idx)
    except Exception as exc:
        logger.warning('Marine API 失敗: %s', exc)


def _fetch_past_weather(target_date: date, result: dict) -> None:
    """Open-Meteo Archive から過去の気象を取得。"""
    try:
        resp = requests.get(_ARCHIVE_URL, params={
            'latitude':   _LAT, 'longitude': _LON, 'timezone': _TZ,
            'start_date': target_date.isoformat(),
            'end_date':   target_date.isoformat(),
            'daily': 'windspeed_10m_max,winddirection_10m_dominant',
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()['daily']
        result['wind_ms_max']  = _safe_get(data.get('windspeed_10m_max', []), 0)
        result['wind_dir_deg'] = _safe_get(data.get('winddirection_10m_dominant', []), 0)
    except Exception as exc:
        logger.warning('Archive 気象 API 失敗: %s', exc)


def _fetch_precip_history(target_date: date, result: dict) -> None:
    """D-1〜D-3 の雨量を Archive から取得。"""
    start = target_date - timedelta(days=3)
    end   = target_date - timedelta(days=1)
    try:
        resp = requests.get(_ARCHIVE_URL, params={
            'latitude':   _LAT, 'longitude': _LON, 'timezone': _TZ,
            'start_date': start.isoformat(),
            'end_date':   end.isoformat(),
            'daily': 'precipitation_sum',
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()['daily']
        dates = [pd.to_datetime(t).date() for t in data.get('time', [])]
        precips = data.get('precipitation_sum', [])
        precip_map = dict(zip(dates, precips))
        for lag in (1, 2, 3):
            d = target_date - timedelta(days=lag)
            result[f'precip_{lag}d'] = (
                float(precip_map[d]) if d in precip_map and precip_map[d] is not None else None
            )
    except Exception as exc:
        logger.warning('Archive 雨量 API 失敗: %s', exc)


# ------------------------------------------------------------------
# 予測エクスポート
# ------------------------------------------------------------------

PREDICTIONS_PATH = Path('ml/predictions.json')


def export_predictions(days: int = 7) -> Path:
    """向こう N 日間の予測を ml/predictions.json に書き出す。

    dashboard.py はこのファイルを読み込むことで、
    モデルや外部APIなしで予測を表示できる。

    Returns:
        書き込んだファイルのパス
    """
    from datetime import datetime
    preds = predict_multi_days(days=days)

    def _serialize(obj):
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj

    serializable = []
    for p in preds:
        serializable.append({k: _serialize(v) for k, v in p.items()})

    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'predictions':  serializable,
    }
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    logger.info('予測エクスポート完了: %s (%d日分)', PREDICTIONS_PATH, len(preds))
    return PREDICTIONS_PATH


# ------------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------------

def _safe_get(lst: list, idx: int):
    try:
        v = lst[idx]
        return float(v) if v is not None else None
    except (IndexError, TypeError):
        return None


def _align_categories(X: pd.DataFrame, model) -> None:
    """LightGBM のカテゴリ列を学習時と同じ dtype に揃える。"""
    for col in X.select_dtypes(include='category').columns:
        X.loc[:, col] = X[col].astype('category')


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    if not models_exist():
        print('モデルが見つかりません。先に python ml/train.py を実行してください。')
        sys.exit(1)

    target = date.today() + timedelta(days=1)
    print(f'\n=== {target} の釣行予測 ===')
    result = predict_day(target)
    if result:
        print(f'  期待釣果数 : {result.get("expected_count", "N/A")} 匹')
        print(f'  推奨スコア : {result.get("go_proba", 0) * 100:.1f} %')
        sp = result.get('species_proba', {})
        if sp:
            print('  魚種別確率 :',
                  '  '.join(f'{k}: {v*100:.0f}%' for k, v in sp.items()))

    print('\n=== 磯別ランキング（期待釣果数） ===')
    ranking = predict_spot_ranking(target)
    print(ranking.head(10).to_string(index=False))
