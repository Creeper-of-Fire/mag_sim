"""
运行时状态 - 批处理运行状态
不持久化，仅内存
"""
from __future__ import annotations
from typing import Callable


class RuntimeStore:
    """运行时状态管理（单例）"""

    _instance: RuntimeStore | None = None

    def __new__(cls) -> RuntimeStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._running: bool = False
        self._listeners: list[Callable[[bool], None]] = []

    # ── 读写 ──

    @property
    def is_running(self) -> bool:
        return self._running

    def set_running(self, value: bool):
        """设置运行状态"""
        self._running = value
        self._notify(value)

    # ── 订阅 ──

    def subscribe(self, callback: Callable[[bool], None]):
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[bool], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self, running: bool):
        for listener in self._listeners:
            try:
                listener(running)
            except Exception:
                pass


runtime_store = RuntimeStore()