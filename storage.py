"""
storage.py — CSV書き込み・重複排除
"""

import hashlib
import logging
from datetime import date
from pathlib import Path

import pandas as pd


def _hash_angler(name: str | None) -> str | None:
    """釣り人名をSHA-256ハッシュ（先頭8文字）に変換する。"""
    if not name or (isinstance(name, float)):
        return None
    return hashlib.sha256(str(name).encode('utf-8')).hexdigest()[:8]

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('output')

# CSV の列順（CLAUDE.md のデータモデルに準拠）
COLUMNS = [
    'date', 'day_of_week',
    'weather', 'water_temp_raw', 'water_temp_avg', 'wave_height_raw', 'wave_height_m',
    'next_day_status',
    'spot', 'angler',
    'species_raw', 'species', 'count', 'size_min_cm', 'size_max_cm',
    'tackle', 'bait',
    'detail_url',
]

# 重複排除キー
DEDUP_KEYS = ['date', 'spot', 'angler', 'species']


def save(entries: list[dict]) -> Path:
    """釣果エントリをCSVに追記保存する（重複排除付き）。

    ファイル名: output/YYYYMMDD_rockshore.csv（実行日付）
    既存ファイルがあれば読み込み、新規エントリとマージしてから書き直す。

    Returns:
        書き込んだCSVのパス
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    today_str = date.today().strftime('%Y%m%d')
    csv_path  = OUTPUT_DIR / f'{today_str}_rockshore.csv'

    new_df = pd.DataFrame(entries)
    # 未定義列を None で補完し、列順を統一
    for col in COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[COLUMNS]

    # 釣り人名を匿名化（SHA-256先頭8文字）
    new_df['angler'] = new_df['angler'].apply(_hash_angler)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str)
        before = len(existing_df)
        combined = pd.concat([existing_df, new_df.astype(str)], ignore_index=True)
        combined.drop_duplicates(subset=DEDUP_KEYS, keep='last', inplace=True)
        logger.info('重複排除: %d + %d → %d 件', before, len(new_df), len(combined))
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return csv_path
