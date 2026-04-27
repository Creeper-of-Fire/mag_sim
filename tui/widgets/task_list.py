# task_list.py - 更新导入
"""
任务列表组件：读取 tasks.csv，显示任务状态，支持选择事件
"""
import csv
import json
import hashlib
from pathlib import Path
from textual.widgets import Static, ListView, ListItem
from textual.containers import Vertical
from textual import on
from textual.message import Message

from utils.project_config import (
    FILENAME_TASKS_CSV,
    FILENAME_HISTORY,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    COLUMN_TASK_NAME
)


class TaskList(Vertical):
    """任务列表面板"""

    class TaskSelected(Message):
        """任务被选中时发送的消息"""

        def __init__(self, task_data: dict, task_index: int):
            super().__init__()
            self.task_data = task_data
            self.task_index = task_index

    def compose(self):
        yield Static("📋 任务列表", classes="panel_title")
        yield ListView(id="list_view")

    def on_mount(self):
        """监听 ListView 的选择变化"""
        self._tasks = []

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected):
        """当列表项被选中时，发送任务详情"""
        if event.item is None:
            return

        # 获取选中项的索引
        list_view = self.query_one("#list_view", ListView)
        idx = list_view.index
        if idx is not None and 0 <= idx < len(self._tasks):
            task = self._tasks[idx]
            self.post_message(self.TaskSelected(task, idx))

    def load_from_dir(self, job_dir: Path):
        """从 tasks.csv 和 history.jsonl 加载任务并渲染"""
        list_view = self.query_one("#list_view", ListView)
        list_view.clear()
        self._tasks = []

        csv_path = job_dir / FILENAME_TASKS_CSV
        history_path = job_dir / FILENAME_HISTORY

        if not csv_path.exists():
            list_view.append(ListItem(Static(f"（无 {FILENAME_TASKS_CSV}）")))
            return

        # 读取 CSV
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self._tasks = [row for row in reader if any(row.values())]

        # 读取历史记录
        history_status = {}
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        h = entry.get("hash", "")
                        status = STATUS_COMPLETED if entry.get("status") == "success" else STATUS_FAILED
                        history_status[h] = status
                    except (json.JSONDecodeError, KeyError):
                        continue

        # 构建列表项
        for task in self._tasks:
            name = task.get(COLUMN_TASK_NAME, "unnamed")
            # 计算哈希
            params = {k: v for k, v in task.items() if k != COLUMN_TASK_NAME}
            p_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
            p_hash = hashlib.sha256(p_str.encode()).hexdigest()[:12]

            status = history_status.get(p_hash, STATUS_PENDING)
            label = f" [{status}] {name}"
            item = ListItem(Static(label))
            list_view.append(item)

    def get_current_task(self):
        """获取当前选中的任务"""
        list_view = self.query_one("#list_view", ListView)
        idx = list_view.index
        if idx is not None and 0 <= idx < len(self._tasks):
            return self._tasks[idx]
        return None