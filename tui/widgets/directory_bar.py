"""
顶部信息栏：显示当前工作目录
"""
from pathlib import Path
from textual.widgets import Static
from textual.containers import Horizontal


class DirectoryBar(Horizontal):
    """显示当前选择的项目目录"""

    def compose(self):
        yield Static("📁 项目目录: ", id="dir_label", classes="dir_bar_label")
        yield Static("（未选择）", id="dir_path", classes="dir_bar_path")

    def update_path(self, path: Path):
        """更新显示的路径"""
        self.query_one("#dir_path", Static).update(str(path))