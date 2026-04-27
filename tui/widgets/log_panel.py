"""
日志输出面板：带滚动和自动追底
"""
from textual.binding import Binding, BindingsMap
from textual.containers import Vertical
from textual.widgets import Static, TextArea

from tui.store.log_store import logger


class LogTextArea(TextArea):
    """日志专用 TextArea，只保留基础只读操作"""
    BINDINGS = [
        Binding("up", "cursor_up", show=False),
        Binding("down", "cursor_down", show=False),
        Binding("left", "cursor_left", show=False),
        Binding("right", "cursor_right", show=False),
        Binding("pageup", "cursor_page_up", show=False),
        Binding("pagedown", "cursor_page_down", show=False),
        Binding("ctrl+a", "select_all", "全选", show=True),
        Binding("ctrl+c,super+c", "copy", show=False),
    ]

    def on_mount(self):
        # 移除 TextArea 的所有默认绑定
        text_area = self
        text_area._bindings = BindingsMap()  # 清空所有绑定

        # 只添加你需要的
        text_area._bindings.bind("up", "cursor_up", show=False)
        text_area._bindings.bind("down", "cursor_down", show=False)
        text_area._bindings.bind("pageup", "cursor_page_up", show=False)
        text_area._bindings.bind("pagedown", "cursor_page_down", show=False)
        text_area._bindings.bind("ctrl+a", "select_all", "全选", show=True)
        text_area._bindings.bind("ctrl+c,super+c", "copy","复制", show=False)

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
        Binding("ctrl+a", "select_all", "全选", priority=True, show=False),
    ]

    def compose(self):
        yield Static("📜 运行日志", classes="panel_title")
        yield LogTextArea(
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
            text_area = self.query_one("#log_content", LogTextArea)
            text_area.text = "\n".join(existing)
            text_area.scroll_to(y=len(text_area.text.splitlines()) - 1, animate=False)

        # 订阅新日志
        logger.subscribe(self._on_log_received)

    def on_unmount(self):
        """卸载时取消订阅"""
        logger.unsubscribe(self._on_log_received)

    def _on_log_received(self, line: str):
        """收到新日志时更新显示"""
        text_area = self.query_one("#log_content", LogTextArea)
        current = text_area.text
        new_text = f"{current}\n{line}" if current else line
        text_area.text = new_text
        text_area.scroll_to(y=len(text_area.text.splitlines()) - 1, animate=False)

    def clear(self):
        """清空日志（同时清空 store）"""
        logger.clear()
        try:
            self.query_one("#log_content", LogTextArea).text = ""
        except Exception:
            pass

    def action_select_all(self):
        """全选日志内容"""
        try:
            self.query_one("#log_content", LogTextArea).select_all()
        except Exception:
            pass
