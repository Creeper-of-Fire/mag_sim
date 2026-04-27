"""
全局状态管理器 - 保存/恢复 TUI 应用状态
"""
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from utils.project_config import PROJECT_ROOT


@dataclass
class AppState:
    """TUI 全局状态"""
    last_job_dir: str = ""


class StateManager:
    """管理应用状态的持久化"""

    STATE_FILE = PROJECT_ROOT / "data" / "gui_state.json"

    def __init__(self):
        self._state = AppState()

    def load(self) -> AppState:
        """加载状态文件，返回 AppState"""
        if not self.STATE_FILE.exists():
            return self._state

        try:
            with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._state = AppState(
                last_job_dir=data.get("last_job_dir", "")
            )
            return self._state
        except (json.JSONDecodeError, FileNotFoundError):
            return self._state

    def save(self, job_dir: Path | str | None) -> None:
        """保存当前项目目录"""
        if job_dir is None:
            self._state.last_job_dir = ""
        else:
            self._state.last_job_dir = str(job_dir)

        # 确保 data 目录存在
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(self.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self._state), f, indent=2)

    @property
    def last_job_dir(self) -> str:
        return self._state.last_job_dir