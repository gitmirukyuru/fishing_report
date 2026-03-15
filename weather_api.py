"""
weather_api.py — Open-Meteo API ラッパー
APIキー不要・商用無料。通信エラー時は None を返す。
"""

import logging
from datetime import date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# 串本町付近（観測地点）
_LAT = 33.4833
_LON = 135.7833
_TZ  = 'Asia/Tokyo'

_FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
_MARINE_URL   = 'https://marine-api.open-meteo.com/v1/marine'
_ARCHIVE_URL  = 'https://archive-api.open-meteo.com/v1/archive'

_BASE_PARAMS = {'latitude': _LAT, 'longitude': _LON, 'timezone': _TZ}

_DIRECTIONS = ['北', '北北東', '北東', '東北東', '東', '東南東', '南東', '南南東',
               '南', '南南西', '南西', '西南西', '西', '西北西', '北西', '北北西']

_WMO_CODES: dict[int, str] = {
    0: '晴れ', 1: '概ね晴れ', 2: '一部曇り', 3: '曇り',
    45: '霧', 48: '霧氷',
    51: '霧雨（弱）', 53: '霧雨', 55: '霧雨（強）',
    61: '雨（弱）', 63: '雨', 65: '雨（強）',
    71: '雪（弱）', 73: '雪', 75: '雪（強）',
    77: '霰', 80: 'にわか雨（弱）', 81: 'にわか雨', 82: 'にわか雨（強）',
    95: '雷雨', 96: '雷雨（雹）', 99: '激しい雷雨',
}


def deg_to_direction(deg: float) -> str:
    """風向き角度（度）を日本語16方位に変換。"""
    return _DIRECTIONS[round(deg / 22.5) % 16]


def wmo_to_text(code) -> str:
    """WMO天気コードを日本語テキストに変換。"""
    if pd.isna(code):
        return ''
    return _WMO_CODES.get(int(code), f'コード{int(code)}')


def _get_json(url: str, params: dict) -> dict | None:
    """GETリクエストを送りJSONを返す。失敗時はNone。"""
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning('API取得失敗 %s: %s', url, exc)
        return None


def get_forecast() -> pd.DataFrame | None:
    """向こう7日間の天気予報を DataFrame で返す。

    大気予報（天気コード・最高気温）と海洋予報（波高）を結合する。

    Returns:
        columns: date, forecast_weather_code, forecast_wave_height_m, forecast_temp_max
        両API失敗時は None、片方失敗時は取得できた列のみ返す
    """
    atmo = _get_json(_FORECAST_URL, {
        **_BASE_PARAMS,
        'daily': 'weathercode,temperature_2m_max',
        'forecast_days': 7,
    })
    marine = _get_json(_MARINE_URL, {
        **_BASE_PARAMS,
        'daily': 'wave_height_max',
        'forecast_days': 7,
    })

    if atmo is None and marine is None:
        return None

    dfs = []
    if atmo:
        d = atmo['daily']
        dfs.append(pd.DataFrame({
            'date':                  d['time'],
            'forecast_weather_code': d.get('weathercode'),
            'forecast_temp_max':     d.get('temperature_2m_max'),
        }))
    if marine:
        d = marine['daily']
        dfs.append(pd.DataFrame({
            'date':                   d['time'],
            'forecast_wave_height_m': d.get('wave_height_max'),
        }))

    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = dfs[0].merge(dfs[1], on='date', how='outer')

    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def get_hourly_forecast(days: int = 14) -> pd.DataFrame | None:
    """向こうN日間の6時間ごと天気詳細を DataFrame で返す。

    Returns:
        columns: date, hour, temp_c, precipitation_mm,
                 wind_speed_ms, wind_dir_deg, wind_dir, weather_code, weather_text
        失敗時は None
    """
    data = _get_json(_FORECAST_URL, {
        **_BASE_PARAMS,
        'hourly': 'temperature_2m,precipitation,windspeed_10m,winddirection_10m,weathercode',
        'forecast_days': days,
        'wind_speed_unit': 'ms',
    })
    if data is None:
        return None

    h = data['hourly']
    df = pd.DataFrame({
        'datetime':        pd.to_datetime(h['time']),
        'temp_c':          h.get('temperature_2m'),
        'precipitation_mm': h.get('precipitation'),
        'wind_speed_ms':   h.get('windspeed_10m'),
        'wind_dir_deg':    h.get('winddirection_10m'),
        'weather_code':    h.get('weathercode'),
    })

    df = df[df['datetime'].dt.hour.isin([0, 6, 12, 18])].copy()
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['wind_dir'] = df['wind_dir_deg'].apply(
        lambda d: deg_to_direction(d) if pd.notna(d) else ''
    )
    df['weather_text'] = df['weather_code'].apply(wmo_to_text)
    return df.reset_index(drop=True)


def get_historical(start_date: date, end_date: date) -> pd.DataFrame | None:
    """指定期間の過去気象データを DataFrame で返す。

    Returns:
        columns: date, forecast_weather_code, forecast_wave_height_m
        通信・パースエラー時は None
    """
    date_params = {
        'start_date': start_date.isoformat(),
        'end_date':   end_date.isoformat(),
    }
    atmo = _get_json(_ARCHIVE_URL, {
        **_BASE_PARAMS, **date_params,
        'daily': 'weathercode',
    })
    marine = _get_json(_MARINE_URL, {
        **_BASE_PARAMS, **date_params,
        'daily': 'wave_height_max',
    })

    if atmo is None and marine is None:
        return None

    dfs = []
    if atmo:
        d = atmo['daily']
        dfs.append(pd.DataFrame({
            'date':                  d['time'],
            'forecast_weather_code': d.get('weathercode'),
        }))
    if marine:
        d = marine['daily']
        dfs.append(pd.DataFrame({
            'date':                   d['time'],
            'forecast_wave_height_m': d.get('wave_height_max'),
        }))

    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = dfs[0].merge(dfs[1], on='date', how='outer')

    df['date'] = pd.to_datetime(df['date']).dt.date
    return df
