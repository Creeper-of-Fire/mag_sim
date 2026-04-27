"""
Textual 应用入口，管理 Screen 切换和全局状态
"""
from pathlib import Path
from textual.app import App

from tui.screens.main_screen import MainScreen


class SimulationTUI(App):
    """等离子体模拟任务管理器"""

    # 加载外部 CSS 文件
    CSS_PATH = Path(__file__).parent / "app.tcss"

    AUTO_FOCUS = None

    # 全局快捷键（在所有 Screen 都生效）
    BINDINGS = [
        ("ctrl+q", "quit", "退出"),
    ]

    def __init__(self):
        super().__init__()

    def on_mount(self):
        """应用启动后进入主界面"""
        self.push_screen(MainScreen())