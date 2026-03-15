"""
parser.py — 1ページ分のHTMLから全釣果エントリを抽出する
"""

import html as html_module
import logging
import re
from datetime import date

logger = logging.getLogger(__name__)

# 魚種正規化マッピング（完全一致）
SPECIES_MAP = {
    'イサギ': 'イサキ',
    '串本': 'グレ',
}

# 部分一致で正規化するパターン（順番に評価、最初に一致したものを使用）
_SPECIES_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'グレ'), 'グレ'),
    (re.compile(r'イサキ|イサギ'), 'イサキ'),
    (re.compile(r'石鯛|サンバソウ'), '石鯛'),
]


def _normalize_species(name: str) -> str:
    """魚種名を正規化する。部分一致パターンを優先し、次にSPECIES_MAPを参照。"""
    for pattern, normalized in _SPECIES_PATTERNS:
        if pattern.search(name):
            return normalized
    return SPECIES_MAP.get(name, name)

# 正規表現パターン
_DATE_RE = re.compile(r'(\d{1,2})月(\d{1,2})日[（(]([月火水木金土日])[)）]')
_COUNT_RE = re.compile(r'(\d+)匹')
_SIZE_RANGE_RE = re.compile(r'(\d+(?:\.\d+)?)cm[〜～](\d+(?:\.\d+)?)cm')
_SIZE_SINGLE_RE = re.compile(r'(\d+(?:\.\d+)?)cm')
_TEMP_RE = re.compile(r'(\d+(?:[.,]\d+)?)[℃度]')
_WAVE_M_RE = re.compile(r'(\d+(?:\.\d+)?)\s*m')


def parse_page(html: str, ref_date: date | None = None) -> list[dict]:
    """ページHTMLを解析し、釣果エントリのリストを返す。

    Args:
        html:     ページ全体のHTML文字列
        ref_date: 年推定の基準日（省略時は今日）

    Returns:
        釣果エントリの辞書リスト
    """
    if ref_date is None:
        ref_date = date.today()

    # <h3> タグで分割（h3タグ自体とその後のコンテンツを交互に取得）
    parts = re.split(r'(<h3[^>]*>.*?</h3>)', html, flags=re.DOTALL | re.IGNORECASE)

    entries: list[dict] = []

    # parts = [before_first_h3, h3_1, content_1, h3_2, content_2, ...]
    for i in range(1, len(parts), 2):
        h3_html = parts[i]
        content_html = parts[i + 1] if i + 1 < len(parts) else ''

        date_info = _parse_date(h3_html, ref_date)
        if not date_info:
            logger.debug('日付なしh3をスキップ: %.60s', h3_html)
            continue

        meta = _parse_meta(content_html)
        rows = _parse_table(content_html)
        # detail_url は h3 内の <a href> が優先、なければコンテンツ内を探す
        detail_url = _parse_h3_url(h3_html) or _parse_detail_url(content_html)

        if not rows:
            # 休船日などテーブルなしエントリ
            entries.append({
                **date_info, **meta,
                'spot': None, 'angler': None,
                'species_raw': None, 'species': None,
                'count': None, 'size_min_cm': None, 'size_max_cm': None,
                'tackle': None, 'bait': None,
                'detail_url': detail_url,
            })
        else:
            for row in rows:
                entries.append({**date_info, **meta, **row, 'detail_url': detail_url})

    return entries


# ---------------------------------------------------------------------------
# 内部パース関数
# ---------------------------------------------------------------------------

def _parse_date(h3_html: str, ref_date: date) -> dict | None:
    """h3 HTMLから日付・曜日を抽出する。"""
    m = _DATE_RE.search(h3_html)
    if not m:
        return None

    month, day, day_of_week = int(m.group(1)), int(m.group(2)), m.group(3)
    year = _infer_year(month, day, ref_date)

    try:
        d = date(year, month, day)
    except ValueError:
        logger.warning('無効な日付: %d-%02d-%02d', year, month, day)
        return None

    return {'date': d.isoformat(), 'day_of_week': day_of_week}


def _infer_year(month: int, day: int, ref_date: date) -> int:
    """月・日から年を推定する（未来日付なら前年とみなす）。"""
    year = ref_date.year
    try:
        if date(year, month, day) > ref_date:
            return year - 1
    except ValueError:
        pass
    return year


def _parse_meta(content: str) -> dict:
    """天候・水温・波高・翌日情報を抽出する。"""
    # タグを改行に置換してプレーンテキスト化
    text = re.sub(r'<[^>]+>', '\n', content)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text).strip()

    weather = _extract_labeled(text, '天候')
    water_temp_raw = _extract_labeled(text, '水温')
    wave_height_raw = _extract_labeled(text, '波高')

    # 翌日情報：明日/翌日/出船予定/休船 を含む行
    next_day_status: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if re.search(r'明日|翌日|休船|出船予定', line):
            next_day_status = line
            break

    water_temp_avg = _calc_water_temp_avg(water_temp_raw) if water_temp_raw else None

    # 波高の数値（先頭の数値を m 単位で抽出）
    wave_height_m: float | None = None
    if wave_height_raw:
        wm = _WAVE_M_RE.search(wave_height_raw)
        wave_height_m = float(wm.group(1)) if wm else None

    return {
        'weather': weather,
        'water_temp_raw': water_temp_raw,
        'water_temp_avg': water_temp_avg,
        'wave_height_raw': wave_height_raw,
        'wave_height_m': wave_height_m,
        'next_day_status': next_day_status,
    }


def _extract_labeled(text: str, label: str) -> str | None:
    """'ラベル：値' 形式の値を行単位で抽出する。"""
    m = re.search(rf'{label}[：:]\s*(.+)', text)
    return m.group(1).strip() if m else None


def _calc_water_temp_avg(water_temp_raw: str) -> float | None:
    """水温文字列から数値の平均を計算する。"""
    temps = _TEMP_RE.findall(water_temp_raw)
    if not temps:
        return None
    return round(sum(float(t.replace(',', '.')) for t in temps) / len(temps), 2)


_MULTI_SEP_RE = re.compile(r'[、,，]\s*')

def _split_species_text(text: str) -> list[str]:
    """複数魚種テキストを個別の魚種テキストに分割する。

    例:
        "コルダイ1匹、石鯛2匹、サンバソウ1匹" → ["コルダイ1匹", "石鯛2匹", "サンバソウ1匹"]
        "グレ6匹30cm〜37cmオオモンハタ1匹" → ["グレ6匹30cm〜37cm", "オオモンハタ1匹"]
        "グレ1匹36cm オオモンハタ1匹" → ["グレ1匹36cm", "オオモンハタ1匹"]
    """
    # Step 1: 読点・コンマで分割
    parts = [p.strip() for p in _MULTI_SEP_RE.split(text) if p.strip()]

    # Step 2: 同一部分に複数「匹」が含まれる場合、cm境界でさらに分割
    result = []
    for part in parts:
        if part.count('匹') <= 1:
            result.append(part)
        else:
            # cm直後（または cm + スペース後）に日本語文字が続く位置で分割
            sep = re.sub(r'cm(?=[ \t]*[^\d〜～\.、,，\s\(（])', 'cm\x00', part)
            sub = [p.strip() for p in sep.split('\x00') if p.strip()]
            result.extend(sub if len(sub) > 1 else [part])

    return result


def _parse_table(content_html: str) -> list[dict]:
    """釣果テーブル（2列：左=ラベル<br>区切り、右=値<br>区切り）を解析する。

    実際のHTML構造:
        <tr>
          <td class="td_name">釣り場<br>釣り人<br>魚種<br>仕掛け<br>エサ</td>
          <td class="td_value">沖の長島<br>藤井様<br>グレ8匹30cm〜40.5cm<br>フカセ<br>オキアミ</td>
        </tr>
    """
    table_m = re.search(r'<table[^>]*>(.*?)</table>', content_html, re.DOTALL | re.IGNORECASE)
    if not table_m:
        return []

    all_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_m.group(1), re.DOTALL | re.IGNORECASE)

    rows: list[dict] = []
    for tr in all_rows:
        tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL | re.IGNORECASE)
        if len(tds) < 2:
            continue

        # <br> で分割してラベルと値を対応付ける
        labels = [_strip_tags(x).strip() for x in re.split(r'<br\s*/?>', tds[0], flags=re.IGNORECASE)]
        values = [_strip_tags(x).strip() for x in re.split(r'<br\s*/?>', tds[1], flags=re.IGNORECASE)]

        base_entry: dict = {}
        species_raw_text: str | None = None

        for label, value in zip(labels, values):
            label = label.strip()
            value = value.strip() or None
            if label == '釣り場':
                base_entry['spot'] = value
            elif label == '釣り人':
                base_entry['angler'] = value
            elif label == '魚種':
                species_raw_text = value
            elif label == '仕掛け':
                base_entry['tackle'] = value
            elif label == 'エサ':
                base_entry['bait'] = value

        if not base_entry:
            continue

        if not species_raw_text:
            base_entry.update({
                'species_raw': None, 'species': None,
                'count': None, 'size_min_cm': None, 'size_max_cm': None,
            })
            rows.append(base_entry)
        else:
            for sp_text in _split_species_text(species_raw_text):
                e = base_entry.copy()
                e.update(_parse_species(sp_text))
                rows.append(e)

    return rows


def _parse_species(species_raw: str) -> dict:
    """魚種文字列から魚種名・匹数・サイズを抽出する。

    例: '串本11匹32cm〜40.2cm' → species='グレ', count=11, size_min=32.0, size_max=40.2
    """
    count_m = _COUNT_RE.search(species_raw)
    count = int(count_m.group(1)) if count_m else None

    range_m = _SIZE_RANGE_RE.search(species_raw)
    if range_m:
        size_min: float | None = float(range_m.group(1))
        size_max: float | None = float(range_m.group(2))
    else:
        single_m = _SIZE_SINGLE_RE.search(species_raw)
        if single_m:
            size_min = size_max = float(single_m.group(1))
        else:
            size_min = size_max = None

    # 先頭の非数字部分を魚種名として取得（数字始まりは魚種名なしとみなす）
    name_m = re.match(r'^([^\d]+)', species_raw)
    name = name_m.group(1).strip() if name_m else None
    species = _normalize_species(name) if name else None

    return {
        'species_raw': species_raw,
        'species':     species,
        'count':       count,
        'size_min_cm': size_min,
        'size_max_cm': size_max,
    }


def _parse_h3_url(h3_html: str) -> str | None:
    """h3 内の <a href> からURLを抽出する（詳細ページあり日付エントリに存在）。"""
    m = re.search(r'<a[^>]+href=["\']([^"\']+)["\']', h3_html, re.IGNORECASE)
    return html_module.unescape(m.group(1)) if m else None


def _parse_detail_url(content_html: str) -> str | None:
    """コンテンツ内の「詳細を見る」リンクのURLを抽出する。"""
    m = re.search(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>\s*詳細を見る\s*</a>',
        content_html, re.IGNORECASE
    )
    return html_module.unescape(m.group(1)) if m else None


def _strip_tags(html: str) -> str:
    """HTMLタグを除去してプレーンテキストを返す。"""
    return re.sub(r'<[^>]+>', '', html).strip()
