"""
应用全局状态
持久化到 gui_state.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from utils.project_config import PROJECT_ROOT


class AppState(BaseModel):
    """持久化的应用状态"""
    last_job_dir: str = Field(
        default="",
        description="上次打开的模拟项目目录路径"
    )
    theme: str = Field(
        default="plasma-dark",
        description="当前主题名称",
        pattern=r"^plasma-(dark|light)$"
    )


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
        self._state: AppState = self._load_state()
        self._listeners: list[Callable[[AppState], None]] = []

    # ── 读取 ──

    @property
    def theme(self) -> str:
        return self._state.theme

    @property
    def job_dir(self) -> Path | None:
        path_str = self._state.last_job_dir
        if not path_str:
            return None
        path = Path(path_str)
        return path if path.exists() and path.is_dir() else None

    # ── 写入 ──

    def set_theme(self, theme: str, persist: bool = True):
        self._state.theme = theme
        if persist:
            self._save_state()
        self._notify()

    def set_job_dir(self, path: Path, persist: bool = True):
        self._state.last_job_dir = str(path)
        if persist:
            self._save_state()
        self._notify()

    # ── 持久化 ──

    def _load_state(self) -> AppState:
        if not self.STATE_FILE.exists():
            return AppState()
        try:
            data = json.loads(self.STATE_FILE.read_text(encoding="utf-8"))
            return AppState(**data)
        except (json.JSONDecodeError, FileNotFoundError):
            return AppState()

    def _save_state(self):
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.STATE_FILE.write_text(
            self._state.model_dump_json(indent=2),
            encoding="utf-8"
        )

    # ── 订阅 ──

    def subscribe(self, callback: Callable[[AppState], None]):
        """订阅状态变化，回调参数为 (key, value)"""
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[AppState], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self):
        for listener in self._listeners:
            try:
                listener(self._state)
            except Exception:
                pass

app_store = AppStore()
