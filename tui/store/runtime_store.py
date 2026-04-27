"""
运行时状态 - 批处理运行状态
不持久化，仅内存
"""
from __future__ import annotations
from typing import Callable

from pydantic import BaseModel


class RuntimeState(BaseModel):
    """运行时状态"""
    is_running: bool = False

class RuntimeStore:
    """运行时状态管理（单例）"""

    _instance: RuntimeStore | None = None

    def __new__(cls) -> RuntimeStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._state: RuntimeState = RuntimeState()
        self._listeners: list[Callable[[RuntimeState], None]] = []

    # ── 读写 ──

    @property
    def is_running(self) -> bool:
        return self._state.is_running

    def set_running(self, value: bool):
        """设置运行状态"""
        self._state.is_running = value
        self._notify()

    # ── 订阅 ──

    def subscribe(self, callback: Callable[[RuntimeState], None]):
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[RuntimeState], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self):
        for listener in self._listeners:
            try:
                listener(self._state)
            except Exception:
                pass

runtime_store = RuntimeStore()