"""
顶部信息栏：显示当前工作目录
"""
from pathlib import Path

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Static, Button

from tui.store.app_store import app_store, AppState
from tui.store.log_store import logger


class DirectoryBar(Horizontal):
    """显示当前选择的项目目录"""

    DEFAULT_CSS = """
    DirectoryBar {
        background: $bg-secondary;
        color: $text-primary;
        padding: 0 1;
        height: 3;
        max-height: 3;
        align: center middle;
        border: solid $border-primary;
    }

    DirectoryBar:focus {
        background: $bg-tertiary;
        border: solid $border-focus;
    }

    #dir_label {
        color: $text-muted;
        width: 14;
    }

    #dir_path {
        color: $text-accent;
        width: 1fr;
    }
    """

    can_focus = True

    BINDINGS = [
        Binding("enter", "select_dir", "选择目录", priority=True),
        Binding("f", "open_in_fm", "打开文件夹", priority=True),
    ]

    def compose(self):
        yield Static("📁 项目目录: ", id="dir_label")
        yield Static("（未选择 | Enter选择 F打开）", id="dir_path")
        yield Button("选择目录", id="btn_select_dir", variant="primary")
        yield Button("打开文件夹", id="btn_open_fm", variant="warning")

    def on_mount(self):
        """获取焦点提示"""
        self.border_title = "[Enter]选择 [F]打开"
        # 订阅 AppStore 变化
        app_store.subscribe(self._on_dir_changed)
        # 恢复已有目录
        if app_store.job_dir:
            self._update_display(app_store.job_dir)

    def on_unmount(self):
        app_store.unsubscribe(self._on_dir_changed)

    def _on_dir_changed(self, state: AppState):
        """AppStore 目录变化时更新显示"""
        self._update_display(Path(state.last_job_dir))

    def _update_display(self, path: Path):
        """更新路径显示"""
        self.query_one("#dir_path", Static).update(str(path))

    # ── 按键动作 ──

    @on(Button.Pressed, "#btn_select_dir")
    def action_select_dir(self):
        """打开目录选择对话框"""
        from tui.screens.file_dialog import open_directory_dialog

        initial = str(app_store.job_dir) if app_store.job_dir else str(Path.home())

        path_str = open_directory_dialog(
            title="选择等离子体模拟项目目录",
            initial_dir=initial
        )
        if not path_str:
            return

        path = Path(path_str)
        if not path.exists() or not path.is_dir():
            logger.error(f"错误: 无效目录 {path}")
            return

        app_store.set_job_dir(path)
        logger.info(f"日志: 已切换到 {path}")

    @on(Button.Pressed, "#btn_open_fm")
    def action_open_in_fm(self):
        """在文件管理器中打开"""
        import os
        import sys
        import subprocess as sp

        if not app_store.job_dir:
            logger.warn("警告: 请先选择目录")
            return

        path = app_store.job_dir
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            sp.run(["open", str(path)])
        else:
            sp.run(["xdg-open", str(path)])

        logger.info(f"日志: 已打开 {path.name}")
