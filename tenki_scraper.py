"""
tenki_scraper.py — tenki.jp 14日間天気予報 + tide736.net 潮汐データ取得
"""

import json
import logging
import re
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from scrapling.fetchers import Fetcher

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 設定
# -------------------------------------------------------------------

FORECAST_URLS = {
    '串本': 'https://tenki.jp/forecast/6/33/6520/30428/10days.html',
    '白浜': 'https://tenki.jp/forecast/6/33/6520/30401/10days.html',
}

HOURLY_URLS = {
    '串本': 'https://tenki.jp/forecast/6/33/6520/30428/1hour.html',
    '白浜': 'https://tenki.jp/forecast/6/33/6520/30401/1hour.html',
}

# tide736.net 港コード（pc=都道府県コード, hc=港コード）
TIDE_PORTS = {
    '串本': {'pc': '30', 'hc': '3'},
    '白浜': {'pc': '30', 'hc': '13'},  # 田辺（白浜最寄港）
}

_TIDE_API = 'https://api.tide736.net/get_tide.php'


# -------------------------------------------------------------------
# 公開関数
# -------------------------------------------------------------------

def get_forecast_wind_risk(location: str) -> pd.DataFrame | None:
    """tenki.jp 10days.html の6時間ごと風速から朝方リスクを返す（4日目以降用）。

    各日の wind-item ブロック（<dd class="wind-item">）内の
    <span>Xm/s</span> を先頭2スロット（0:00・6:00）だけ取得し最大値で判定。

    Returns:
        columns: date, wind_max_ms, risk_label, risk_prob
        失敗時は None
    """
    url = FORECAST_URLS.get(location)
    if not url:
        logger.warning('未知のロケーション: %s', location)
        return None
    try:
        page = Fetcher().get(url)
        html = str(page.html_content)
        return _parse_forecast_wind_risk(html)
    except Exception as exc:
        logger.warning('10days wind取得失敗 (%s): %s', location, exc)
        return None


def _parse_forecast_wind_risk(html: str) -> pd.DataFrame | None:
    """10days.html HTML から日別朝方風速を抽出してリスクラベルを返す。"""
    today = date.today()
    blocks = re.split(r'(\d{1,2}月\d{1,2}日)', html)

    rows = []
    for j in range(1, len(blocks), 2):
        date_str = blocks[j]
        content  = blocks[j + 1] if j + 1 < len(blocks) else ''

        dm = re.match(r'(\d{1,2})月(\d{1,2})日', date_str)
        if not dm:
            continue
        month, day = int(dm.group(1)), int(dm.group(2))
        try:
            d = date(today.year, month, day)
            if d < today - timedelta(days=1):
                d = date(today.year + 1, month, day)
        except ValueError:
            continue

        # wind-item ブロックを抽出
        wind_block = re.search(
            r'<dd[^>]*class="wind-item"[^>]*>([\s\S]*?)</dd>', content
        )
        if not wind_block:
            continue

        # 各スロットの風速（m/s）を抽出
        speeds = [
            int(m.group(1))
            for m in re.finditer(r'<span>(\d+)m/s</span>', wind_block.group(1))
        ]
        if not speeds:
            continue

        # 先頭2スロット（0:00・6:00）の最大を朝方風速とする
        morning_max = max(speeds[:2]) if len(speeds) >= 2 else speeds[0]

        if morning_max >= 6:
            label, prob = '休船確率大', 0.90
        elif morning_max >= 5:
            label, prob = '休船可能性あり', 0.75
        else:
            label, prob = '出船可能', 0.0

        rows.append({
            'date':        d,
            'wind_max_ms': float(morning_max),
            'risk_label':  label,
            'risk_prob':   prob,
        })

    return pd.DataFrame(rows) if rows else None


def get_morning_wind(location: str) -> pd.DataFrame | None:
    """tenki.jp 1hour.html から今日〜明後日の 0〜6時 最大風速を取得。

    Returns:
        columns: date, wind_max_ms, risk_label, risk_prob
            risk_label: '出船可能' / '休船可能性あり' / '休船確率大'
            risk_prob:  0.0 / 0.75 / 0.90
        失敗時は None
    """
    url = HOURLY_URLS.get(location)
    if not url:
        logger.warning('未知のロケーション: %s', location)
        return None
    try:
        page = Fetcher().get(url)
        html = str(page.html_content)
        return _parse_morning_wind(html)
    except Exception as exc:
        logger.warning('1時間予報取得失敗 (%s): %s', location, exc)
        return None


def _parse_morning_wind(html: str) -> pd.DataFrame | None:
    """1hour.html を解析して日付×0〜6時最大風速を返す。"""
    hour_rows  = re.findall(r'<tr[^>]*class="hour"[^>]*>([\s\S]*?)</tr>', html)
    wind_rows  = re.findall(r'<tr[^>]*class="wind-speed"[^>]*>([\s\S]*?)</tr>', html)
    if not hour_rows or not wind_rows:
        return None

    def extract_tds(row_html):
        tds = re.findall(r'<td[^>]*>([\s\S]*?)</td>', row_html)
        return [re.sub(r'<[^>]+>', '', t).strip() for t in tds]

    today = date.today()
    rows = []
    for i, (hr, wr) in enumerate(zip(hour_rows, wind_rows)):
        target_date = today + timedelta(days=i)
        hours  = extract_tds(hr)
        speeds = extract_tds(wr)
        morning_max = None
        for h_str, s_str in zip(hours, speeds):
            try:
                h = int(h_str)
                s = int(s_str)
                if 1 <= h <= 6:
                    morning_max = max(morning_max, s) if morning_max is not None else s
            except (ValueError, TypeError):
                continue

        if morning_max is None:
            continue

        if morning_max >= 6:
            label, prob = '休船確率大', 0.90
        elif morning_max >= 5:
            label, prob = '休船可能性あり', 0.75
        else:
            label, prob = '出船可能', 0.0

        rows.append({
            'date':         target_date,
            'wind_max_ms':  morning_max,
            'risk_label':   label,
            'risk_prob':    prob,
        })

    return pd.DataFrame(rows) if rows else None


def get_weather_forecast(location: str) -> pd.DataFrame | None:
    """tenki.jp 10days.html から14日間の6時間ごと天気予報を取得。

    Returns:
        columns: datetime, date, hour, temp_c, humidity,
                 weather, temp_max, temp_min, wind, sunrise, sunset
        失敗時は None
    """
    url = FORECAST_URLS.get(location)
    if not url:
        logger.warning('未知のロケーション: %s', location)
        return None

    try:
        page = Fetcher().get(url)
        html = str(page.html_content)
        return _parse_forecast(html)
    except Exception as exc:
        logger.warning('天気予報取得失敗 (%s): %s', location, exc)
        return None


def get_tide(location: str, days: int = 14) -> pd.DataFrame | None:
    """tide736.net APIから満潮・干潮データを取得。

    Returns:
        columns: date, type（満潮/干潮）, time, height_cm, tide_name
        失敗時は None
    """
    port = TIDE_PORTS.get(location)
    if not port:
        logger.warning('未知のロケーション: %s', location)
        return None

    rows = []
    today = date.today()

    for i in range(days):
        target = today + timedelta(days=i)
        try:
            resp = requests.get(_TIDE_API, params={
                **port,
                'yr': str(target.year),
                'mn': str(target.month),
                'dy': str(target.day),
                'rg': 'day',
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            chart = data.get('tide', {}).get('chart', {})
            day_data = chart.get(target.isoformat(), {})
            tide_name = day_data.get('moon', {}).get('title', '')

            for entry in day_data.get('flood', []):
                rows.append({
                    'date': target, 'type': '満潮',
                    'time': entry['time'], 'height_cm': entry['cm'],
                    'tide_name': tide_name,
                })
            for entry in day_data.get('edd', []):
                rows.append({
                    'date': target, 'type': '干潮',
                    'time': entry['time'], 'height_cm': entry['cm'],
                    'tide_name': tide_name,
                })

        except Exception as exc:
            logger.warning('潮汐取得失敗 %s %s: %s', location, target, exc)

        if i < days - 1:
            time.sleep(0.3)

    return pd.DataFrame(rows) if rows else None


def get_sun_times(location: str, days: int = 14) -> pd.DataFrame | None:
    """tide736.net APIから日の出・日の入り時刻を取得。

    Returns:
        columns: date, sunrise, sunset
    """
    port = TIDE_PORTS.get(location)
    if not port:
        return None

    rows = []
    today = date.today()

    for i in range(days):
        target = today + timedelta(days=i)
        try:
            resp = requests.get(_TIDE_API, params={
                **port,
                'yr': str(target.year),
                'mn': str(target.month),
                'dy': str(target.day),
                'rg': 'day',
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            sun = (data.get('tide', {})
                       .get('chart', {})
                       .get(target.isoformat(), {})
                       .get('sun', {}))
            rows.append({
                'date':    target,
                'sunrise': sun.get('rise', ''),
                'sunset':  sun.get('set', ''),
            })
        except Exception as exc:
            logger.warning('日の出取得失敗 %s %s: %s', location, target, exc)

        if i < days - 1:
            time.sleep(0.3)

    return pd.DataFrame(rows) if rows else None


# -------------------------------------------------------------------
# 内部パース関数
# -------------------------------------------------------------------

def _parse_forecast(html: str) -> pd.DataFrame | None:
    """10days.html HTML を解析して DataFrame を返す。"""
    dataset = _extract_json_ld_dataset(html)
    if not dataset:
        logger.warning('JSON-LD Dataset が見つかりません')
        return None

    table = dataset.get('mainEntity', {})
    columns = table.get('csvw:tableSchema', {}).get('csvw:columns', [])

    col_map: dict[str, list[str]] = {}
    for col in columns:
        name = col.get('csvw:name', '')
        col_map[name] = [c.get('csvw:value', '') for c in col.get('csvw:cells', [])]

    if '日時' not in col_map:
        return None

    rows = []
    for i, dt_str in enumerate(col_map['日時']):
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', dt_str)
        if not m:
            continue
        dt = datetime.fromisoformat(m.group(1))
        if dt.hour not in (0, 6, 12, 18):
            continue

        rows.append({
            'datetime': dt,
            'date':     dt.date(),
            'hour':     dt.hour,
            'temp_c':   _safe_float(col_map.get('気温(℃)', []), i),
            'humidity': _safe_float(col_map.get('湿度(％)', []), i),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # 日別データ（天気・最高最低・風）をHTMLからパースしてmerge
    daily = _parse_daily(html)
    if daily:
        daily_df = (pd.DataFrame(daily)
                      .drop_duplicates(subset=['date'], keep='first'))
        df = df.merge(daily_df, on='date', how='left')

    return df


def _extract_json_ld_dataset(html: str) -> dict | None:
    """HTML から気象データ用 Dataset JSON-LD を抽出。

    '#10days' を含む @id を優先。なければ '日時' 列を持つ Dataset を返す。
    """
    candidates: list[dict] = []
    for s in re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.DOTALL | re.IGNORECASE
    ):
        try:
            data = json.loads(s.strip())
            for item in (data if isinstance(data, list) else [data]):
                if isinstance(item, dict) and item.get('@type') == 'Dataset':
                    candidates.append(item)
        except Exception:
            continue

    # '@id' に '#10days' が含まれるものを優先
    for item in candidates:
        if '#10days' in item.get('@id', ''):
            return item

    # 次善: '日時' 列を持つものを返す
    for item in candidates:
        cols = (item.get('mainEntity', {})
                    .get('csvw:tableSchema', {})
                    .get('csvw:columns', []))
        if any(c.get('csvw:name') == '日時' for c in cols):
            return item

    return None


def _parse_daily(html: str) -> list[dict]:
    """HTMLから日別の天気・最高最低気温・風・日の出日の入りを抽出。

    HTML構造（tenki.jp 10days.html）:
      <span class="high-temp">13℃</span><span class="low-temp">5℃</span>
      <span class="forecast-telop">曇</span>
      日の出｜06:10 / 日の入｜18:03
    """
    results: list[dict] = []
    today = date.today()

    # 日付テキスト（曜日なし）で分割
    blocks = re.split(r'(\d{1,2}月\d{1,2}日)', html)

    for j in range(1, len(blocks), 2):
        date_str = blocks[j]
        content  = blocks[j + 1] if j + 1 < len(blocks) else ''

        dm = re.match(r'(\d{1,2})月(\d{1,2})日', date_str)
        if not dm:
            continue
        month, day = int(dm.group(1)), int(dm.group(2))
        try:
            d = date(today.year, month, day)
            if d < today - timedelta(days=1):
                d = date(today.year + 1, month, day)
        except ValueError:
            continue

        # 最高気温がないブロックは副次的なブロック（レーダー等）のためスキップ
        hm = re.search(r'class="high-temp">(\d+)℃', content)
        if not hm:
            continue

        entry: dict = {'date': d}

        # 最高・最低気温
        lm = re.search(r'class="low-temp">(\d+)℃', content)
        entry['temp_max'] = int(hm.group(1))
        if lm:
            entry['temp_min'] = int(lm.group(1))

        # 天気テキスト（forecast-telop クラス優先）
        wm = re.search(r'class="forecast-telop">([^<]+)', content)
        if wm:
            entry['weather'] = wm.group(1).strip()

        # 日の出・日の入り
        sun_m    = re.search(r'日の出[｜|](\d{2}:\d{2})', content)
        sunset_m = re.search(r'日の入[｜|](\d{2}:\d{2})', content)
        if sun_m:
            entry['sunrise'] = sun_m.group(1)
        if sunset_m:
            entry['sunset'] = sunset_m.group(1)

        # 風速（"北の風 3m/s" 形式）
        wnd = re.search(r'([北南東西][北南東西]*)の?風\s*(\d+(?:\.\d+)?)\s*m', content)
        if wnd:
            entry['wind'] = f'{wnd.group(1)} {wnd.group(2)}m/s'

        results.append(entry)

    return results


def _safe_float(lst: list[str], i: int) -> float | None:
    try:
        return float(lst[i]) if i < len(lst) else None
    except (ValueError, TypeError):
        return None
