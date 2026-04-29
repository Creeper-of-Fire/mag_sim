"""pyte 终端仿真器 — 在 Textual 中渲染 ANSI 输出"""
from __future__ import annotations

import threading
from io import StringIO

import pyte
from textual.widgets import RichLog
from textual.containers import Vertical

# pyte 颜色名 → Rich 颜色名
_PYTE_TO_RICH_COLOR = {
    "black": "#1a1a1a",
    "red": "#e06c75",
    "green": "#98c379",
    "yellow": "#e5c07b",
    "blue": "#61afef",
    "magenta": "#c678dd",
    "cyan": "#56b6c2",
    "white": "#abb2bf",
    "brightblack": "#5c6370",
    "brightred": "#e06c75",
    "brightgreen": "#98c379",
    "brightyellow": "#e5c07b",
    "brightblue": "#61afef",
    "brightmagenta": "#c678dd",
    "brightcyan": "#56b6c2",
    "brightwhite": "#ffffff",
}


def _pyte_to_rich(buffer_row, width: int) -> str:
    """将一行 pyte buffer 转换为 Rich markup 字符串"""
    cur_fg = cur_bg = cur_bold = cur_italic = cur_underscore = None

    def _style():
        tags = []
        if cur_fg:
            tags.append(cur_fg)
        if cur_bg:
            tags.append(cur_bg)
        if cur_bold:
            tags.append("b")
        if cur_italic:
            tags.append("i")
        if cur_underscore:
            tags.append("u")
        return tags

    def _open(tags):
        return "".join(f"[{t}]" for t in tags)

    def _close(tags):
        return "".join(f"[/{t.split()[0].split('=')[0]}]" for t in reversed(tags))

    result = ""
    old_tags: list[str] = []

    for x in range(width):
        char = buffer_row[x]  # StaticDefaultDict: 缺失列返回默认空格 Char
        ch = char.data
        fg = char.fg if char.fg != "default" else None
        bg = char.bg if char.bg != "default" else None
        bold = char.bold
        italic = char.italics
        underscore = char.underscore
        if ch == "\x00":
            ch = " "

        new_fg = f'#{fg}' if fg and fg.startswith("4") and len(fg) == 6 else \
                 _PYTE_TO_RICH_COLOR.get(fg) if fg else None
        new_bg = f"on #{bg}" if bg and bg.startswith("4") and len(bg) == 6 else \
                 f"on {_PYTE_TO_RICH_COLOR.get(bg)}" if bg and bg in _PYTE_TO_RICH_COLOR else None

        tags = _style()
        new_tags = [t for t in tags if t is not None]

        if new_tags != old_tags:
            result += _close(old_tags)
            result += _open(new_tags)
            old_tags = new_tags

        result += ch.replace("[", "\\[")

    result += _close(old_tags)
    return result.rstrip()


class TerminalWidget(Vertical):
    """终端仿真器面板"""

    CSS = """
    TerminalWidget {
        border: solid $border-primary;
        background: #0d1017;
    }

    #term_display {
        height: 1fr;
        background: #0d1017;
        color: #abb2bf;
    }
    """

    def __init__(self, rows: int = 40, cols: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._rows = rows
        self._cols = cols
        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.Stream(self._screen)
        self._lock = threading.Lock()
        self._pending = ""
        self._dirty = False

    def compose(self):
        yield RichLog(
            id="term_display",
            highlight=False,
            markup=True,
            auto_scroll=True,
            max_lines=2000,
        )

    def on_mount(self):
        self.set_interval(0.05, self._refresh_display)

    # ── 线程安全 API ──

    def feed(self, text: str):
        """线程安全：喂 ANSI 文本到终端"""
        with self._lock:
            self._stream.feed(text)
            self._dirty = True

    def write(self, text: str):
        """兼容 sys.stdout.write() 的接口"""
        self.feed(text)

    def flush(self):
        pass

    def clear_screen(self):
        with self._lock:
            self._screen.reset()
            self._dirty = True

    @property
    def display(self) -> RichLog:
        return self.query_one("#term_display", RichLog)

    # ── 渲染 ──

    def _refresh_display(self):
        if not self._dirty:
            return
        with self._lock:
            self._dirty = False
            lines_to_render = list(self._screen.display)

        log = self.display
        log.clear()
        last_nonempty = -1
        for i, line in enumerate(lines_to_render):
            if line.strip():
                last_nonempty = i
        for i, line in enumerate(lines_to_render):
            if i > last_nonempty:
                break
            row = self._screen.buffer[i]  # StaticDefaultDict[int, Char]
            rich = _pyte_to_rich(row, self._cols)
            if rich:
                log.write(rich)

    # ── stdout 重定向 ──

    def redirect_stdout(self):
        """上下文管理器，把 sys.stdout 重定向到本终端"""
        return _StdoutRedirect(self)


class _StdoutRedirect:
    def __init__(self, term: TerminalWidget):
        import sys as _sys
        self._term = term
        self._sys = _sys
        self._saved = None

    def __enter__(self):
        self._saved = self._sys.stdout
        self._sys.stdout = self._term
        return self

    def __exit__(self, *args):
        self._sys.stdout = self._saved
