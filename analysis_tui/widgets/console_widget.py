"""嵌入式 Rich 控制台 — RichLog + stdout 桥接

把后台线程中的 console.print() / print() 输出捕获到 TUI 的 RichLog。"""
from __future__ import annotations

import sys
import threading
from io import StringIO
from typing import Callable

from textual.containers import Vertical
from textual.widgets import RichLog, Static

from analysis.core.utils import console as rich_console


class _StdoutRedirector:
    """线程安全的 stdout 拦截器。每行文本推送给回调。"""

    def __init__(self, callback: Callable[[str], None]):
        self._callback = callback
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, s: str):
        with self._lock:
            self._buffer += s
            if "\n" in self._buffer:
                lines = self._buffer.split("\n")
                self._buffer = lines.pop()  # 最后一段是不完整的行
                for line in lines:
                    if line:
                        self._callback(line)

    def flush(self):
        with self._lock:
            if self._buffer:
                self._callback(self._buffer)
                self._buffer = ""

    def isatty(self) -> bool:
        return False


class ConsoleWidget(Vertical):
    """控制台面板：带标题的 RichLog"""

    CSS = """
    ConsoleWidget {
        border: solid $border-primary;
        background: $bg-primary;
    }

    #console_log {
        height: 1fr;
        min-height: 5;
        background: $bg-primary;
    }
    """

    def compose(self):
        yield Static("— 控制台", classes="panel_title")
        yield RichLog(id="console_log", highlight=True, markup=True, auto_scroll=True)

    @property
    def _log(self) -> RichLog:
        return self.query_one("#console_log", RichLog)

    def write(self, text: str):
        """线程安全：向 RichLog 写入 Rich markup 文本。可在线程池中直接调用。"""
        try:
            self.app.call_from_thread(self._log.write, text)
        except Exception:
            pass

    def clear(self):
        """清空控制台"""
        try:
            self._log.clear()
        except Exception:
            pass

    def redirect_stdout(self):
        """返回一个上下文管理器，在线程内重定向 sys.stdout → 本控制台。"""
        return _StdoutRedirectContext(self.write)


class _StdoutRedirectContext:
    """上下文管理器：在 with 块内把 sys.stdout 重定向到指定回调。"""

    def __init__(self, callback: Callable[[str], None]):
        self._callback = callback
        self._redirector = _StdoutRedirector(callback)
        self._original_stdout = None
        self._original_console_file = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self._redirector
        # Rich 的 console 对象也重定向
        self._original_console_file = rich_console.file
        rich_console.file = self._redirector
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        rich_console.file = self._original_console_file
        self._redirector.flush()
