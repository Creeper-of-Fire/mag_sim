"""
日志输出面板：带滚动和自动追底
"""
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static, TextArea

from tui.store.log_store import logger


class LogPanel(Vertical):
    """日志面板"""

    CSS = """
    LogPanel {
        border: solid #0f3460;
        background: #1a1a2e;
    }

    #log_content {
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
        color: #c0c0c0;
        background: #1a1a2e;
    }
    """

    BINDINGS = [
        Binding("ctrl+a", "select_all", "全选", priority=True),
    ]

    def compose(self):
        yield Static("📜 运行日志", classes="panel_title")
        yield TextArea(
            id="log_content",
            read_only=True,
            soft_wrap=True,
            text=""
        )

    def on_mount(self):
        """挂载时订阅 LogStore，加载已有日志"""
        # 加载现有日志
        existing = logger.all
        if existing:
            text_area = self.query_one("#log_content", TextArea)
            text_area.text = "\n".join(existing)
            text_area.scroll_to(len(text_area.text), 0, animate=False)

        # 订阅新日志
        logger.subscribe(self._on_log_received)

    def on_unmount(self):
        """卸载时取消订阅"""
        logger.unsubscribe(self._on_log_received)

    def _on_log_received(self, line: str):
        """收到新日志时更新显示"""
        try:
            text_area = self.query_one("#log_content", TextArea)
            current = text_area.text
            new_text = f"{current}\n{line}" if current else line
            text_area.text = new_text
            text_area.scroll_to(len(new_text), 0, animate=False)
        except Exception:
            pass  # 组件可能已销毁

    def clear(self):
        """清空日志（同时清空 store）"""
        logger.clear()
        try:
            self.query_one("#log_content", TextArea).text = ""
        except Exception:
            pass

    def action_select_all(self):
        """全选日志内容"""
        try:
            self.query_one("#log_content", TextArea).select_all()
        except Exception:
            pass
