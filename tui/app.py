"""
Textual 应用入口，管理 Screen 切换和全局状态
"""
from pathlib import Path

from textual.app import App
from textual.theme import Theme

from tui.screens.main_screen import MainScreen
from tui.store.app_store import app_store


class SimulationTUI(App):
    """等离子体模拟任务管理器"""

    AUTO_FOCUS = None

    CSS_PATH = [
        Path(__file__).parent / "app.tcss",
    ]

    BINDINGS = [
        ("ctrl+q", "quit", "退出"),
        ("ctrl+t", "toggle_theme", "切换主题"),
    ]

    def __init__(self):
        super().__init__()
        self._register_themes()
        self.theme = app_store.theme

    def _register_themes(self):
        # 注册深色主题
        self.register_theme(
            Theme(
                name="plasma-dark",
                primary="#1a1a2e",
                secondary="#16213e",
                accent="#00ff88",
                background="#1a1a2e",
                surface="#16213e",
                panel="#0f3460",
                success="#00ff88",
                warning="#ffaa00",
                error="#ff4444",
                variables={
                    "bg-primary": "#1a1a2e",
                    "bg-secondary": "#16213e",
                    "bg-tertiary": "#0f3460",
                    "bg-input": "#1a1a2e",
                    "text-primary": "#e0e0e0",
                    "text-secondary": "#c0c0c0",
                    "text-muted": "#888888",
                    "text-accent": "#00ff88",
                    "border-primary": "#0f3460",
                    "border-focus": "#e94560",
                    "highlight-bg": "#0f3460",
                    "highlight-text": "#ffffff",
                    "modal-overlay": "rgba(0, 0, 0, 0.6)",
                    "color-success": "#00ff88",
                    "color-warning": "#ffaa00",
                    "color-error": "#ff4444",
                }
            )
        )
        # 注册浅色主题
        self.register_theme(
            Theme(
                name="plasma-light",
                primary="#ffffff",
                secondary="#f0f2f5",
                accent="#2a6e3f",
                background="#f0f2f5",
                surface="#ffffff",
                panel="#e0e5ec",
                success="#2a6e3f",
                warning="#cc8800",
                error="#cc2222",
                variables={
                    "bg-primary": "#f0f2f5",
                    "bg-secondary": "#ffffff",
                    "bg-tertiary": "#e0e5ec",
                    "bg-input": "#ffffff",
                    "text-primary": "#1a1a2e",
                    "text-secondary": "#333333",
                    "text-muted": "#666666",
                    "text-accent": "#2a6e3f",
                    "border-primary": "#cccccc",
                    "border-focus": "#e94560",
                    "highlight-bg": "#d0d8e8",
                    "highlight-text": "#1a1a2e",
                    "modal-overlay": "rgba(0, 0, 0, 0.3)",
                    "color-success": "#2a6e3f",
                    "color-warning": "#cc8800",
                    "color-error": "#cc2222",
                }
            )
        )

    def on_mount(self):
        """设置默认主题并显示主界面"""
        self.push_screen(MainScreen())

    def action_toggle_theme(self):
        """切换白天/黑夜模式"""
        new_theme = "plasma-light" if self.theme == "plasma-dark" else "plasma-dark"
        self.theme = new_theme
        app_store.set_theme(new_theme)
