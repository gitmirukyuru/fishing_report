"""
scraper.py — エントリポイント・ページネーション制御・状態管理
"""

import argparse
import json
import logging
import random
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

from scrapling.fetchers import Fetcher

import parser as page_parser
import storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

BASE_URL   = 'https://www.taniguchitosen.com/rockshore'
STATE_FILE = Path('state.json')
MAX_PAGES  = 60


# ---------------------------------------------------------------------------
# 状態管理
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    return {}


def save_state(last_date: str) -> None:
    STATE_FILE.write_text(
        json.dumps({'last_date': last_date}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


# ---------------------------------------------------------------------------
# フェッチ
# ---------------------------------------------------------------------------

def _page_url(page: int) -> str:
    return BASE_URL if page == 1 else f'{BASE_URL}/page/{page}'


def _fetch_html(fetcher: Fetcher, url: str) -> str:
    """ページを取得してHTML文字列を返す。"""
    response = fetcher.get(url)
    # html_content は TextHandler（str互換）を返す
    html = str(response.html_content)
    if not html:
        raise ValueError(f'空のレスポンス: {url}')
    return html


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def run(cutoff_date: date | None, full: bool) -> None:
    """スクレイピングを実行する。

    Args:
        cutoff_date: この日付より古いエントリは取得しない（None = 制限なし）
        full:        True のとき全60ページを強制取得
    """
    fetcher = Fetcher()
    state   = load_state()
    today   = date.today()

    # インクリメンタルモードの cutoff を state.json から読み込む
    if not full and cutoff_date is None:
        last_str = state.get('last_date')
        if last_str:
            cutoff_date = date.fromisoformat(last_str)
            logger.info('前回取得日: %s 以降を取得します', last_str)

    all_entries: list[dict] = []
    newest_date: date | None = None

    for page_num in range(1, MAX_PAGES + 1):
        url = _page_url(page_num)
        logger.info('ページ %d 取得: %s', page_num, url)

        try:
            html = _fetch_html(fetcher, url)
        except Exception as exc:
            logger.error('取得失敗 %s: %s', url, exc)
            break

        entries = page_parser.parse_page(html, ref_date=today)
        if not entries:
            logger.info('ページ %d にエントリなし。終了します。', page_num)
            break

        reached_cutoff = False
        for entry in entries:
            entry_date = date.fromisoformat(entry['date'])

            if newest_date is None or entry_date > newest_date:
                newest_date = entry_date

            if cutoff_date and entry_date <= cutoff_date:
                reached_cutoff = True
                break  # このエントリ以降は古い

            all_entries.append(entry)

        if reached_cutoff and not full:
            logger.info('カットオフ日 %s に到達。終了します。', cutoff_date)
            break

        if page_num < MAX_PAGES:
            wait = random.uniform(1.0, 3.0)
            logger.debug('%.1f 秒待機...', wait)
            time.sleep(wait)

    if all_entries:
        out_path = storage.save(all_entries)
        logger.info('%d 件を保存しました → %s', len(all_entries), out_path)

        # 新規データがある場合のみモデルを再学習
        logger.info('モデルを再学習します...')
        result = subprocess.run(
            ['python', '-m', 'ml.train'],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            logger.info('モデル再学習完了')
        else:
            logger.warning('モデル再学習失敗:\n%s', result.stderr[-500:])
    else:
        logger.info('新規エントリなし。モデル再学習をスキップします。')

    if newest_date:
        save_state(newest_date.isoformat())
        logger.info('state.json を更新: last_date = %s', newest_date)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='谷口渡船 磯釣り釣果スクレイパー',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python scraper.py              # インクリメンタル取得（state.json の日付以降）
  python scraper.py --days 7     # 直近7日分取得
  python scraper.py --full       # 全60ページ取得
""",
    )
    ap.add_argument('--full', action='store_true', help='全60ページを取得する')
    ap.add_argument('--days', type=int, metavar='N', help='直近N日分を取得する')
    args = ap.parse_args()

    cutoff: date | None = None
    if args.days is not None:
        cutoff = today = date.today() - timedelta(days=args.days)
        logger.info('--days %d: %s 以降を取得します', args.days, cutoff)

    run(cutoff_date=cutoff, full=args.full)


if __name__ == '__main__':
    main()
