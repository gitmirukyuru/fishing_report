from pathlib import Path
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent
_func = components.declare_component("scroll_reset", path=str(_COMPONENT_DIR))


def scroll_reset(key: str = "scroll_reset") -> None:
    """ダイアログのスクロール位置をトップにリセットする。"""
    _func(key=key, default=0)
