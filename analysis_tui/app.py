"""Textual 应用入口，管理 Screen 切换和主题"""
from pathlib import Path

from textual.app import App
from textual.binding import Binding
from textual.theme import Theme


class AnalysisTUI(App):
    """等离子体模拟分析管理器"""

    AUTO_FOCUS = None

    CSS_PATH = [
        Path(__file__).parent / "app.tcss",
    ]

    TITLE = "Plasma Analysis"

    BINDINGS = [
        Binding("ctrl+Q", "quit", "退出"),
        Binding("ctrl+T,ctrl+t", "toggle_theme", "切换主题"),
    ]

    def __init__(self):
        super().__init__()
        self._register_themes()
        self.theme = "plasma-dark"

    def _register_themes(self):
        self.register_theme(
            Theme(
                name="plasma-dark",
                primary="#1a1a2e",
                secondary="#16213e",
                accent="#00ff88",
                background="#0d1117",
                surface="#161b22",
                panel="#21262d",
                success="#00ff88",
                warning="#ffaa00",
                error="#ff4444",
                variables={
                    "bg-primary": "#0d1117",
                    "bg-secondary": "#161b22",
                    "bg-tertiary": "#21262d",
                    "bg-input": "#0d1117",
                    "text-primary": "#e0e0e0",
                    "text-secondary": "#c0c0c0",
                    "text-muted": "#888888",
                    "text-accent": "#00ff88",
                    "border-primary": "#30363d",
                    "border-focus": "#58a6ff",
                    "highlight-bg": "#1f2428",
                    "highlight-text": "#ffffff",
                    "modal-overlay": "rgba(0, 0, 0, 0.6)",
                    "color-success": "#00ff88",
                    "color-warning": "#ffaa00",
                    "color-error": "#ff4444",
                }
            )
        )
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
        from analysis_tui.screens.analysis_screen import AnalysisScreen
        self.push_screen(AnalysisScreen())

    def action_toggle_theme(self):
        new_theme = "plasma-light" if self.theme == "plasma-dark" else "plasma-dark"
        self.theme = new_theme
