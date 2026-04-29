"""终端 — Rich ANSI → RichLog + 内嵌输入行"""
from __future__ import annotations

import queue
import threading

from rich.text import Text
from textual import on
from textual.widgets import RichLog, Input
from textual.containers import Vertical


class _TermInput(Input):
    """输入行 — 回车写入 RichLog + 推入 stdin 队列"""

    def __init__(self, stdin_queue: queue.Queue, display: RichLog, **kwargs):
        super().__init__(placeholder="", **kwargs)
        self._stdin_queue = stdin_queue
        self._display = display

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted):
        val = event.value or ""
        if val:
            self._display.write(f" [bold #00ff88]{val}[/bold #00ff88]\n")
        self._stdin_queue.put(val)
        self.value = ""


class TerminalWidget(Vertical):
    """终端面板"""

    CSS = """
    TerminalWidget {
        background: $bg-primary;
        border: none;
        padding: 0;
    }

    #term_display {
        height: 1fr;
        background: $bg-primary;
        color: $text-primary;
        padding: 0 1;
        border: none;
        margin: 0;
    }

    #term_input_line {
        height: auto;
        background: $bg-secondary;
        color: $text-primary;
        border: hkey $border-primary;
        padding: 0 1;
        margin: 0;
    }

    #term_input_line:focus {
        border: hkey $border-focus;
        background: $bg-input;
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
        log = RichLog(
            id="term_display",
            highlight=False,
            markup=True,
            auto_scroll=True,
            max_lines=5000,
        )
        yield log
        yield _TermInput(self._stdin_queue, log, id="term_input_line")

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
        self._drain_stdin()
        try:
            self.display.clear()
        except Exception:
            pass

    def _drain_stdin(self):
        while not self._stdin_queue.empty():
            try:
                self._stdin_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def display(self) -> RichLog:
        return self.query_one("#term_display", RichLog)

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
