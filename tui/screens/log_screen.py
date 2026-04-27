"""
简洁日志界面：仅显示目录栏 + 运行日志
"""
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, Footer

from tui.widgets.directory_bar import DirectoryBar
from tui.widgets.log_panel import LogPanel


class LogScreen(Screen):
    """只显示目录和日志的简洁视图"""

    AUTO_FOCUS = None

    CSS = """
    #log_screen_dir_bar {
        height: 3;
    }

    #log_screen_log_panel {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("I,i,escape", "back_to_main", "返回主界面", priority=True)
    ]

    def compose(self):
        yield Header()
        yield DirectoryBar(id="log_screen_dir_bar")
        yield LogPanel(id="log_screen_log_panel")
        yield Footer()

    def action_back_to_main(self):
        self.app.pop_screen()
