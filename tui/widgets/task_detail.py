# task_detail.py
"""
任务详情面板：显示选中任务的参数
"""
from textual.widgets import Static
from textual.containers import Vertical, VerticalScroll


class TaskDetail(Vertical):
    """任务详情显示区"""

    CSS = """
    TaskDetail {
        border: solid #0f3460;
        background: #1a1a2e;
    }

    #detail_scroll {
        height: 1fr;
        overflow-y: auto;
        scrollbar-size: 0 0;
    }

    #detail_content {
        padding: 0 1;
        color: #c0c0c0;
        height: auto;
    }
    """

    def compose(self):
        yield Static("🔍 任务详情", classes="panel_title")
        with VerticalScroll(id="detail_scroll"):
            yield Static("（请选择左侧任务查看详情）", id="detail_content")

    def show_task(self, params: dict):
        """显示任务参数"""
        lines = [f"{k}: {v}" for k, v in params.items()]
        detail = self.query_one("#detail_content", Static)
        detail.update("\n".join(lines))

    def clear(self):
        """清空详情"""
        detail = self.query_one("#detail_content", Static)
        detail.update("（请选择左侧任务查看详情）")