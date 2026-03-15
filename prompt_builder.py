"""
prompt_builder.py — Claude.ai 手動貼り付け用プロンプト生成

ダッシュボードの「Claudeへのプロンプトをコピー」ボタンから呼び出す。
"""

from datetime import date


def build_prompt(
    target_date: date,
    weather: str,
    temp_max: float | None,
    temp_min: float | None,
    wind_dir: str,
    wind_speed_ms: float | None,
    wave_height_m: float | None,
    water_temp_c: float | None,
    tide_name: str,
    rising_ratio: float | None,
    precip_1d: float | None,
    precip_2d: float | None,
    precip_3d: float | None,
    predicted_count: float | None,
    go_score_pct: float | None,
    species_rank: list[tuple[str, float]],
    n_similar: int,
    avg_count_similar: float | None,
) -> str:
    """釣行アドバイス用プロンプト文を生成する。

    Args:
        species_rank:      [(魚種名, 出現確率), ...] 上位3種
        n_similar:         過去の類似条件レコード数
        avg_count_similar: 類似条件での平均釣果数

    Returns:
        Claude.ai に貼り付けるプロンプト文字列
    """
    # 整形ヘルパー
    def _fmt_float(v, fmt='.1f', unit='', fallback='不明') -> str:
        return f'{v:{fmt}}{unit}' if v is not None else fallback

    def _fmt_temp(hi, lo) -> str:
        if hi is not None and lo is not None:
            return f'{lo:.0f}〜{hi:.0f}℃'
        if hi is not None:
            return f'{hi:.0f}℃'
        return '不明'

    species_str = (
        '  '.join(f'{sp}({p*100:.0f}%)' for sp, p in species_rank)
        if species_rank else '不明'
    )

    similar_str = (
        f'{n_similar}件中 平均{avg_count_similar:.1f}匹'
        if n_similar > 0 and avg_count_similar is not None
        else 'データ不足'
    )

    rising_str = (
        f'{rising_ratio*100:.0f}%'
        if rising_ratio is not None else '不明'
    )

    prompt = f"""\
以下の条件での磯釣り（谷口渡船・串本）釣行についてアドバイスをください。

【釣行予定日】{target_date.strftime('%Y年%m月%d日')}
【気象】天気: {weather} / 気温: {_fmt_temp(temp_max, temp_min)} / 風: {wind_dir} {_fmt_float(wind_speed_ms, '.1f', 'm/s')}
【海況】波高: {_fmt_float(wave_height_m, '.1f', 'm')} / 水温: {_fmt_float(water_temp_c, '.1f', '℃')}
【潮汐】潮名: {tide_name or '不明'} / 上り潮割合（7〜14時）: {rising_str}
【雨歴】前日: {_fmt_float(precip_1d, '.1f', 'mm')} / 2日前: {_fmt_float(precip_2d, '.1f', 'mm')} / 3日前: {_fmt_float(precip_3d, '.1f', 'mm')}

【AI予測結果】
- 期待釣果: {_fmt_float(predicted_count, '.1f', '匹（一人あたり）')}
- 釣行推奨スコア: {_fmt_float(go_score_pct, '.1f', '%')}
- 釣れやすい魚種: {species_str}
- 過去類似条件: {similar_str}

推奨する仕掛け・エサ・狙い目の時間帯・磯の選び方を含めて200字程度でアドバイスをください。\
"""
    return prompt
