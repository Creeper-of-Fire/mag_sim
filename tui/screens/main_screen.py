# main_screen.py
"""
主界面：目录栏 + 任务列表 + 详情/日志
"""
import asyncio
from pathlib import Path

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input

from tui.controllers.csv_tool import CsvToolRunner
from tui.controllers.process_controller import BatchProcessController
from tui.screens.config_screen import ConfigScreen
from tui.store.app_store import app_store
from tui.store.config_store import config_store
from tui.store.log_store import logger
from tui.store.runtime_store import runtime_store
from tui.widgets.directory_bar import DirectoryBar
from tui.widgets.log_panel import LogPanel
from tui.widgets.task_detail import TaskDetail
from tui.widgets.task_list import TaskList
from utils.project_config import (
    PROJECT_ROOT,
    FILENAME_TASKS_CSV
)


class InputDialog(Screen):
    """简单的文本输入弹窗"""

    def __init__(self, prompt: str, callback):
        super().__init__()
        self.prompt = prompt
        self.callback = callback

    def compose(self):
        yield Static(self.prompt)
        yield Input(id="dialog_input", placeholder="输入后按 Enter 确认")

    @on(Input.Submitted)
    def on_submit(self, event: Input.Submitted):
        if event.value.strip():
            self.callback(event.value.strip())
            self.dismiss()


class MainScreen(Screen):
    """主操作界面"""

    CSS = """
    /* 面板通用标题 —— 被多个 Widget 引用，放 Screen 层 */
    .panel_title {
        background: #0f3460;
        color: #e0e0e0;
        padding: 0 1;
        text-style: bold;
        height: 1;
    }
    
    #dir_bar {
        height: 3;
    }
    
    #task_detail {
        height: 30%;
    }
    
    #log_panel {
        height: 70%;
    }
    
    #task_list {
        width: 40%;
    }
    """

    BINDINGS = [
        Binding("s", "start_batch", "启动运行", priority=True),
        Binding("k", "stop_batch", "停止运行", priority=True),
        Binding("r", "refresh", "刷新列表", priority=True),
        Binding("c", "open_config", "编辑配置", priority=True),
    ]

    def compose(self):
        """构建 UI 布局"""
        yield Header()

        # 顶部：目录栏
        yield DirectoryBar(id="dir_bar")

        # 主体：左右分栏
        with Horizontal():
            # 左侧：任务列表（占 40%）
            yield TaskList(id="task_list")

            # 右侧：详情 + 日志
            with Vertical():
                yield TaskDetail(id="task_detail")
                yield LogPanel(id="log_panel")

        yield Footer()

    def on_mount(self):
        """组件挂载完成后的初始化"""
        self._controller = BatchProcessController()
        self._csv_tool = CsvToolRunner()

        # 订阅运行时状态
        runtime_store.subscribe(self._on_runtime_changed)

        # 恢复上次目录
        saved = app_store.load_state()
        if saved:
            app_store.set_job_dir(saved, persist=False)  # 已持久化，不重复写
        else:
            sim_jobs = PROJECT_ROOT / "sim_jobs"
            if sim_jobs.exists() and sim_jobs.is_dir():
                app_store.set_job_dir(sim_jobs)

        self.set_focus(None)

    def on_unmount(self):
        runtime_store.unsubscribe(self._on_runtime_changed)

    # ── 事件处理 ────────────────────────────

    @on(TaskList.TaskSelected)
    def _on_task_selected(self, message: TaskList.TaskSelected):
        """处理任务选择事件"""
        task_detail = self.query_one("#task_detail", TaskDetail)
        if message.task_data:
            task_detail.show_task(message.task_data)

    def _on_runtime_changed(self, running: bool):
        """运行时状态变化时更新 UI"""
        try:
            self.query_one("#task_list", TaskList).disabled = running
        except Exception:
            pass

    # ── 内部辅助 ────────────────────────────

    def _get_runner_path(self) -> Path | None:
        """获取 batch_runner.py 的路径"""
        candidates = [
            PROJECT_ROOT / "batch" / "batch_runner.py",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    # ── 快捷键动作 ──────────────────────────

    def action_refresh(self):
        """刷新任务列表"""
        if not app_store.job_dir:
            logger.warn("请先选择项目目录。")
            return
        try:
            task_list = self.query_one("#task_list", TaskList)
            task_list.load_from_dir(app_store.job_dir)
            self.query_one("#task_detail", TaskDetail).clear()
            logger.info("日志: 任务列表已刷新。")
        except Exception as e:
            logger.error(f"刷新失败: {e}")

    def action_start_batch(self):
        if runtime_store.is_running:
            logger.warn("任务已在运行中。")
            return
        if not app_store.job_dir:
            logger.warn("请先选择项目目录。")
            return

        config = config_store.load()
        csv_path = app_store.job_dir / FILENAME_TASKS_CSV
        if not csv_path.exists():
            logger.error(f"{FILENAME_TASKS_CSV} 不存在，请按 t 创建模板。")
            return

        logger.info(">>> 步骤 1: 转换任务清单...")
        if not self._csv_tool.convert_csv(app_store.job_dir, config.script_name, config.extra_args):
            logger.error("CSV 转换失败，中止运行。")
            return

        runner_path = self._get_runner_path()
        if not runner_path:
            logger.error(f"找不到 batch_runner.py，项目根: {PROJECT_ROOT}")
            return

        runtime_store.set_running(True)
        logger.info(">>> 步骤 2: 启动批处理...")
        asyncio.create_task(self._controller.start(app_store.job_dir, runner_path))

    def action_stop_batch(self):
        """停止运行"""
        if not runtime_store.is_running:
            logger.info("提示: 没有正在运行的任务。")
            return

        logger.info("日志: 正在发送停止信号...")
        asyncio.create_task(self._controller.stop())

    def action_open_config(self):
        """打开配置编辑界面"""
        if not app_store.job_dir:
            logger.warn("警告: 请先选择项目目录。")
            return

        self.app.push_screen(ConfigScreen())
