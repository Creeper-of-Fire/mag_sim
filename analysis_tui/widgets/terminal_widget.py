"""终端 — Rich ANSI 渲染 + 内嵌输入行"""
from __future__ import annotations

import queue
import threading

from rich.text import Text
from textual import events, on
from textual.widgets import RichLog, Input
from textual.containers import Vertical


class _TermInput(Input):
    """终端输入行 — 接收焦点后所有按键直通这里"""

    def __init__(self, stdin_queue: queue.Queue, **kwargs):
        super().__init__(placeholder="", **kwargs)
        self._stdin_queue = stdin_queue

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted):
        self._stdin_queue.put(event.value or "")
        self.value = ""


class TerminalWidget(Vertical):
    """终端面板：RichLog 输出 + Input 输入（无边框融为一体）"""

    CSS = """
    TerminalWidget {
        border: solid $border-primary;
        background: #0d1017;
    }

    TerminalWidget:focus {
        border: solid $border-focus;
    }

    #term_display {
        height: 1fr;
        background: #0d1017;
        color: #abb2bf;
    }

    #term_input_line {
        height: auto;
        background: #0d1017;
        color: #abb2bf;
        border: none;
        padding: 0 1;
    }

    #term_input_line:focus {
        border: none;
    }

    #term_input_line:focus > .input--cursor {
        background: #abb2bf;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._ansi_buffer = ""
        self._stdin_queue: queue.Queue = queue.Queue()

    @property
    def stdin_queue(self) -> queue.Queue:
        return self._stdin_queue

    def compose(self):
        yield RichLog(
            id="term_display",
            highlight=False,
            markup=True,
            auto_scroll=True,
            max_lines=5000,
        )
        yield _TermInput(self._stdin_queue, id="term_input_line")

    def on_mount(self):
        self.set_interval(0.05, self._flush_output)

    # ── stdout（工作线程调用）──

    def write(self, text: str):
        with self._lock:
            self._ansi_buffer += text

    def flush(self):
        pass

    def clear_screen(self):
        with self._lock:
            self._ansi_buffer = ""
        try:
            self.display.clear()
        except Exception:
            pass

    @property
    def display(self) -> RichLog:
        return self.query_one("#term_display", RichLog)

    # ── 内部 ──

    def _flush_output(self):
        with self._lock:
            if not self._ansi_buffer:
                return
            ansi = self._ansi_buffer
            self._ansi_buffer = ""

        try:
            rich_text = Text.from_ansi(ansi)
            if rich_text.plain.strip():
                self.display.write(rich_text)
        except Exception:
            if ansi.strip():
                self.display.write(ansi)
