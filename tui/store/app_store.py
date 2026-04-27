"""
应用全局状态 - 当前项目目录
持久化到 gui_state.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

from utils.project_config import PROJECT_ROOT


class AppStore:
    """应用全局状态管理（单例）"""

    _instance: AppStore | None = None
    STATE_FILE = PROJECT_ROOT / "data" / "gui_state.json"

    def __new__(cls) -> AppStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._job_dir: Path | None = None
        self._listeners: list[Callable[[Path], None]] = []

    # ── 读写 ──

    @property
    def job_dir(self) -> Path | None:
        return self._job_dir

    def set_job_dir(self, path: Path, persist: bool = True):
        """设置当前项目目录"""
        self._job_dir = path
        if persist:
            self._save_state()
        self._notify(path)

    # ── 持久化 ──

    def load_state(self) -> Path | None:
        """加载上次的目录"""
        if not self.STATE_FILE.exists():
            return None
        try:
            data = json.loads(self.STATE_FILE.read_text(encoding="utf-8"))
            path = Path(data.get("last_job_dir", ""))
            if path.exists() and path.is_dir():
                self._job_dir = path
                return path
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return None

    def _save_state(self):
        """持久化到文件"""
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"last_job_dir": str(self._job_dir) if self._job_dir else ""}
        self.STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── 订阅 ──

    def subscribe(self, callback: Callable[[Path], None]):
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[Path], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self, path: Path):
        for listener in self._listeners:
            try:
                listener(path)
            except Exception:
                pass


app_store = AppStore()