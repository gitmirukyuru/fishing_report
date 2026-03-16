"""クリッカブルカード — Streamlit カスタムコンポーネント

カードHTMLをiframe内に描画し、クリックイベントをPythonに返す。
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent
_component_func = components.declare_component("clickable_card", path=str(_COMPONENT_DIR))


def clickable_card(html: str, height: int = 84, key: str | None = None) -> bool:
    """クリッカブルなカードを描画する。

    コンポーネントはクリック時に Date.now() を返す。
    セッションステートで「処理済みの値」を追跡し、
    同じ値が再レンダリングで繰り返し来ても True を返さないようにする。

    Returns:
        True if the card was clicked on this run, else False.
    """
    result = _component_func(html=html, height=height, key=key, default=0)

    # セッションステートキー（コンポーネントkeyごとに管理）
    state_key = f"_cc_last_{key}"
    if result != 0 and result != st.session_state.get(state_key, 0):
        st.session_state[state_key] = result
        return True
    return False
