# main_screen.py
"""
主界面：目录栏 + 任务列表 + 详情/日志
"""
import asyncio
import os
import sys
from pathlib import Path
from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input

from tui.screens.file_dialog import open_directory_dialog
from tui.widgets.directory_bar import DirectoryBar
from tui.widgets.log_panel import LogPanel
from tui.widgets.task_detail import TaskDetail
from tui.widgets.task_list import TaskList
from tui.controllers.process_controller import BatchProcessController
from tui.controllers.job_config import JobConfig, JobConfigManager
from tui.controllers.csv_tool import CsvToolRunner
from tui.state_manager import StateManager

from utils.project_config import (
    PROJECT_ROOT,
    FILENAME_TASKS_CSV,
    FILENAME_HISTORY,
    COLUMN_TASK_NAME
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

    BINDINGS = [
        Binding("s", "start_batch", "启动运行", priority=True),
        Binding("k", "stop_batch", "停止运行", priority=True),
        Binding("r", "refresh", "刷新列表", priority=True),
        Binding("o", "open_dir", "选择目录", priority=True),
        Binding("t", "create_template", "初始化模板", priority=True),
        Binding("f", "open_folder", "打开文件夹", priority=True),
    ]

    def compose(self):
        """构建 UI 布局"""
        yield Header()

        # 顶部：目录栏
        yield DirectoryBar()

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
        self._batch_running = False
        self._controller = BatchProcessController(
            on_log=self._add_log,
            on_finished=self._on_batch_finished
        )
        self._csv_tool = CsvToolRunner(on_log=self._add_log)
        self._job_config: JobConfig | None = None
        self._config_manager: JobConfigManager | None = None

        # 恢复上次的目录
        state_mgr = StateManager()
        state = state_mgr.load()
        if state.last_job_dir:
            path = Path(state.last_job_dir)
            if path.exists() and path.is_dir():
                self._switch_to_dir(path, save_state=False)  # 已保存过，不重复写
                return

        # 没有可恢复的目录时，尝试默认打开 sim_jobs
        sim_jobs = PROJECT_ROOT / "sim_jobs"
        if sim_jobs.exists() and sim_jobs.is_dir():
            self._switch_to_dir(sim_jobs, save_state=False)

    # ── 事件处理 ────────────────────────────

    @on(TaskList.TaskSelected)
    def _on_task_selected(self, message: TaskList.TaskSelected):
        """处理任务选择事件"""
        task_detail = self.query_one("#task_detail", TaskDetail)
        if message.task_data:
            task_detail.show_task(message.task_data)

    # ── 内部辅助 ────────────────────────────

    def _add_log(self, message: str) -> None:
        """安全地添加日志行"""
        try:
            log_panel = self._get_log_panel()
            log_panel.add_line(message)
        except Exception:
            pass

    def _get_log_panel(self) -> LogPanel:
        """获取日志面板引用"""
        return self.query_one("#log_panel", LogPanel)

    def _update_ui_state(self):
        """根据运行状态调整 UI"""
        try:
            task_list = self.query_one("#task_list", TaskList)
            task_list.disabled = self._batch_running
        except Exception:
            pass

    def _on_batch_finished(self):
        """批处理完成回调"""
        self._batch_running = False
        try:
            self._update_ui_state()
        except Exception:
            pass
        self._add_log("日志: 批处理运行结束。")
        self.action_refresh()

    def _get_runner_path(self) -> Path | None:
        """获取 batch_runner.py 的路径"""
        candidates = [
            PROJECT_ROOT / "batch" / "batch_runner.py",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _switch_to_dir(self, path: Path, save_state: bool = True):
        """切换到指定项目目录"""
        self.app.current_job_dir = path
        self._config_manager = JobConfigManager(path)
        self._job_config = self._config_manager.load()

        # 更新 UI
        dir_bar = self.query_one(DirectoryBar)
        dir_bar.update_path(path)

        # 保存状态
        if save_state:
            StateManager().save(path)

        self._add_log(f"日志: 已切换到目录 {path}")
        self._add_log(f"日志: 脚本={self._job_config.script_name}, "
                      f"额外参数={self._job_config.extra_args or '(无)'}")
        self.action_refresh()

    # ── 快捷键动作 ──────────────────────────

    def action_open_dir(self):
        """选择项目目录"""
        if self.app.current_job_dir:
            initial = str(self.app.current_job_dir)
        else:
            sim_jobs = PROJECT_ROOT / "sim_jobs"
            initial = str(sim_jobs) if sim_jobs.exists() else str(Path.home())

        path_str = open_directory_dialog(
            title="选择等离子体模拟项目目录",
            initial_dir=initial
        )

        if not path_str:
            return

        path = Path(path_str)
        if not path.exists() or not path.is_dir():
            self._add_log(f"错误: 无效的目录: {path}")
            return

        self._switch_to_dir(path)

    def action_open_folder(self):
        """在文件管理器中打开当前目录"""
        if not self.app.current_job_dir:
            self._add_log("警告: 请先选择项目目录。")
            return

        import subprocess as sp
        job_dir = self.app.current_job_dir
        if sys.platform == "win32":
            os.startfile(job_dir)
        elif sys.platform == "darwin":
            sp.run(["open", str(job_dir)])
        else:
            sp.run(["xdg-open", str(job_dir)])
        self._add_log(f"日志: 已在文件管理器中打开 {job_dir.name}")

    def action_refresh(self):
        """刷新任务列表"""
        job_dir = self.app.current_job_dir
        if not job_dir:
            self._add_log("警告: 请先选择项目目录。")
            return

        try:
            if not job_dir.exists():
                self._add_log(f"错误: 目录不可访问: {job_dir}")
                return
        except Exception as e:
            self._add_log(f"错误: 无法访问目录: {e}")
            return

        task_list = self.query_one("#task_list", TaskList)
        task_list.load_from_dir(job_dir)

        task_detail = self.query_one("#task_detail", TaskDetail)
        task_detail.clear()

        self._add_log("日志: 任务列表已刷新。")

    def action_start_batch(self):
        """开始运行"""
        if self._batch_running:
            self._add_log("警告: 任务已在运行中。")
            return

        if not self.app.current_job_dir:
            self._add_log("警告: 请先选择项目目录。")
            return

        if not self._job_config:
            self._add_log("错误: 项目配置未加载。")
            return

        # 检查 tasks.csv 是否存在
        csv_path = self.app.current_job_dir / FILENAME_TASKS_CSV
        if not csv_path.exists():
            self._add_log(f"错误: {FILENAME_TASKS_CSV} 不存在，请先按 t 创建模板。")
            return

        # 前置步骤：转换 CSV
        self._add_log(">>> 步骤 1: 转换任务清单...")
        if not self._csv_tool.convert_csv(
            self.app.current_job_dir,
            self._job_config.script_name,
            self._job_config.extra_args
        ):
            self._add_log("错误: CSV 转换失败，中止运行。")
            return

        # 获取 runner 路径
        runner_path = self._get_runner_path()
        if not runner_path:
            self._add_log("错误: 找不到 batch_runner.py")
            self._add_log(f"项目根目录: {PROJECT_ROOT}")
            return

        self._batch_running = True
        self._update_ui_state()
        self._add_log(">>> 步骤 2: 启动批处理调度器...")
        self._add_log(f"使用 runner: {runner_path}")

        asyncio.create_task(
            self._controller.start(self.app.current_job_dir, runner_path)
        )

    def action_stop_batch(self):
        """停止运行"""
        if not self._batch_running:
            self._add_log("提示: 没有正在运行的任务。")
            return

        self._add_log("日志: 正在发送停止信号...")
        asyncio.create_task(self._controller.stop())

    def action_create_template(self):
        """初始化模板"""
        if not self.app.current_job_dir:
            self._add_log("警告: 请先选择项目目录。")
            return

        if not self._job_config:
            self._add_log("错误: 项目配置未加载。")
            return

        csv_path = self.app.current_job_dir / FILENAME_TASKS_CSV
        if csv_path.exists():
            self._add_log(f"提示: {FILENAME_TASKS_CSV} 已存在，将覆盖重新生成。")

        # 确保 job_config.json 存在
        self._config_manager.save(self._job_config)

        if self._csv_tool.generate_template(
            self.app.current_job_dir,
            self._job_config.script_name
        ):
            self._add_log(f"日志: 模板已创建，请编辑 {FILENAME_TASKS_CSV} 后刷新。")
            self.action_refresh()
        else:
            self._add_log(f"错误: 模板生成失败。")