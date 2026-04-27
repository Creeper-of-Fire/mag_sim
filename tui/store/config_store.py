"""
项目配置状态 - 脚本名、额外参数
持久化到 {job_dir}/job_config.json
"""
from __future__ import annotations
import json
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Callable

from tui.store.app_store import app_store


class JobConfig(BaseModel):
    """项目运行配置"""
    csv_tool_script: str = Field(default="csv_tool_constant_energy.py", description="CSV 工具的脚本名")
    csv_tool_args: str = Field(default="", description="传给 csv_tool convert 的额外参数")
    runner_args: str = Field(default="", description="传给 batch_runner 的额外参数")


class ConfigStore:
    """项目配置管理（单例）"""

    _instance: ConfigStore | None = None
    FILENAME = "job_config.json"

    def __new__(cls) -> ConfigStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._config: JobConfig = JobConfig()
        self._listeners: list[Callable[[JobConfig], None]] = []

    # ── 读写 ──

    @property
    def config(self) -> JobConfig:
        return self.load()

    def load(self) -> JobConfig:
        """从当前项目目录加载配置，不存在时自动创建默认配置"""
        job_dir = app_store.job_dir
        if not job_dir:
            return JobConfig()

        config_path = job_dir / self.FILENAME
        if not config_path.exists():
            self._config = JobConfig()
            self.save()
            return self._config

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self._config = JobConfig(**data)
        except (json.JSONDecodeError, FileNotFoundError):
            self._config = JobConfig()

        self._notify(self._config)
        return self._config

    def save(self, config: JobConfig | None = None):
        """保存配置到文件"""
        if config is not None:
            self._config = config

        job_dir = app_store.job_dir
        if not job_dir:
            return

        config_path = job_dir / self.FILENAME
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            self._config.model_dump_json(indent=4),
            encoding="utf-8"
        )
        self._notify(self._config)

    @property
    def config_path(self) -> Path | None:
        job_dir = app_store.job_dir
        return job_dir / self.FILENAME if job_dir else None

    # ── 订阅 ──

    def subscribe(self, callback: Callable[[JobConfig], None]):
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[JobConfig], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self, config: JobConfig):
        for listener in self._listeners:
            try:
                listener(config)
            except Exception:
                pass


config_store = ConfigStore()