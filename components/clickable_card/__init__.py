"""クリッカブルカード — Streamlit カスタムコンポーネント

カードHTMLをiframe内に描画し、クリックイベントをPythonに返す。
"""

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent
_component_func = components.declare_component("clickable_card", path=str(_COMPONENT_DIR))


def clickable_card(html: str, height: int = 84, key: str | None = None) -> bool:
    """クリッカブルなカードを描画する。

    Args:
        html: カード内部のHTML文字列（.dcard-wrap を含む）
        height: iframeの高さ (px)
        key: Streamlit の一意キー

    Returns:
        True if the card was clicked on this run, else False.
    """
    result = _component_func(html=html, height=height, key=key, default=0)
    return result != 0
