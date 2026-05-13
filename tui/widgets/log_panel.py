"""
日志输出面板：带滚动和自动追底
"""
from textual.binding import Binding, BindingsMap
from textual.containers import Vertical
from textual.widgets import Static, TextArea

from tui.store.log_store import logger
from tui.widgets.progress_bar import SimulationProgressBar


class LogTextArea(TextArea):
    """日志专用 TextArea，只保留基础只读操作"""

    DEFAULT_CSS = """
    LogTextArea .text-area--selection {
     background: $highlight-bg;
     color: $text-primary;
    }
    """

    def on_mount(self):
        # 移除 TextArea 的所有默认绑定
        text_area = self
        text_area._bindings = BindingsMap()  # 清空所有绑定

        # 只添加你需要的
        text_area._bindings.bind("up", "cursor_up", show=False)
        text_area._bindings.bind("down", "cursor_down", show=False)
        text_area._bindings.bind("pageup", "cursor_page_up", show=False)
        text_area._bindings.bind("pagedown", "cursor_page_down", show=False)
        text_area._bindings.bind("ctrl+C,super+C,ctrl+c,super+c", "copy", "复制", show=False)

class LogPanel(Vertical):
    """日志面板"""

    CSS = """
    LogPanel {
        border: solid $border-primary;
        background: $bg-primary;
    }

    #log_content {
        height: 1fr;
        overflow-y: auto;
        color: $text-secondary;
        background: $bg-primary;
    }
    """

    BINDINGS = [
        Binding("ctrl+A,super+A,ctrl+a,super+a", "select_all", "全选", priority=True, show=True),
    ]

    def compose(self):
        yield Static("📜 运行日志", classes="panel_title")
        yield SimulationProgressBar(id="sim_progress")
        yield LogTextArea(
            id="log_content",
            read_only=True,
            soft_wrap=True,
            text=""
        )

    @property
    def logger_content(self):
        return self.query_one("#log_content", LogTextArea)

    def on_mount(self):
        """挂载时订阅 LogStore，加载已有日志"""
        # 加载现有日志
        existing = logger.all
        if existing:
            text_area = self.logger_content
            text_area.text = "\n".join(existing)
            text_area.scroll_to(y=text_area.document.line_count - 1, animate=False)

        # 订阅新日志
        self._log_buffer: list[str] = []
        logger.subscribe(self._on_log_received)
        self._flush_timer = self.set_interval(0.15, self._flush_log_buffer)

    def on_unmount(self):
        """卸载时取消订阅"""
        logger.unsubscribe(self._on_log_received)

    def _on_log_received(self, line: str):
        """缓冲日志行，不直接更新 UI"""
        self._log_buffer.append(line)

    def _flush_log_buffer(self):
        """定时批量刷新缓冲区到 TextArea"""
        if not self._log_buffer:
            return

        lines = self._log_buffer
        self._log_buffer = []

        text_area = self.logger_content
        current = text_area.text
        text_area.text = f"{current}\n" + "\n".join(lines) if current else "\n".join(lines)
        text_area.scroll_to(y=text_area.document.line_count - 1, animate=False)

    def clear(self):
        """清空日志（同时清空 store）"""
        logger.clear()
        try:
            self.logger_content.text = ""
        except Exception:
            pass  # Widget 可能已卸载

    def action_select_all(self):
        """全选日志内容"""
        try:
            self.logger_content.select_all()
        except Exception:
            pass  # Widget 可能已卸载
