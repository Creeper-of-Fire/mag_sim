"""
日志输出面板：带滚动和自动追底
"""
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static, RichLog, TextArea


class LogPanel(Vertical):
    BINDINGS = [
        Binding("ctrl+a", "select_all", "全选", priority=True),
    ]

    def compose(self):
        yield Static("📜 运行日志", classes="panel_title")
        yield TextArea(
            id="log_content",
            read_only=True,
            soft_wrap=True,
        )

    def add_line(self, text: str):
        """追加一行日志"""
        log = self.query_one("#log_content", TextArea)
        current = log.text
        new_text = f"{current}\n{text}" if current else text
        log.text = new_text
        # 滚动到底部
        log.scroll_to(len(log.text), 0, animate=False)

    def clear(self):
        """清空日志"""
        log = self.query_one("#log_content", TextArea)
        log.text = ""

    def action_select_all(self):
        """全选日志内容"""
        log = self.query_one("#log_content", TextArea)
        log.select_all()