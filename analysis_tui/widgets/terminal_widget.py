"""终端 — Rich Text.from_ansi() 直转 + 键盘直接输入"""
from __future__ import annotations

import queue
import threading

from rich.text import Text
from textual import events
from textual.widgets import RichLog
from textual.containers import Vertical


class TerminalWidget(Vertical):
    """终端：stdout ANSI → Rich Text → RichLog，键盘直接输入"""

    can_focus = True

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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._ansi_buffer = ""
        self._input_buffer = ""
        self.stdin_queue: queue.Queue | None = None

    def compose(self):
        yield RichLog(
            id="term_display",
            highlight=False,
            markup=True,
            auto_scroll=True,
            max_lines=5000,
        )

    def on_mount(self):
        self.set_interval(0.05, self._flush_output)

    # ── 键盘输入 ──

    def on_key(self, event: events.Key) -> None:
        if self.stdin_queue is None:
            return

        key = event.key
        char = event.character

        if key == "enter":
            self.display.write("\n")
            self.stdin_queue.put(self._input_buffer)
            self._input_buffer = ""
            event.stop()
            event.prevent_default()
        elif char and len(char) == 1 and not event.is_forwarded:
            self._input_buffer += char
            self.display.write(char)
            event.stop()
            event.prevent_default()

    # ── stdout（工作线程调用）──

    def write(self, text: str):
        with self._lock:
            self._ansi_buffer += text

    def flush(self):
        pass

    def clear_screen(self):
        with self._lock:
            self._ansi_buffer = ""
            self._input_buffer = ""
        try:
            self.display.clear()
        except Exception:
            pass

    @property
    def display(self) -> RichLog:
        return self.query_one("#term_display", RichLog)

    # ── 内部 ──

    def _flush_output(self):
        """定时刷新 ANSI 缓冲 → Rich Text → RichLog"""
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
