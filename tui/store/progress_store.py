"""
模拟进度状态 - 跟踪 WarpX 模拟的实时进度
不持久化，仅内存
"""
from __future__ import annotations
from typing import Callable

from pydantic import BaseModel


class ProgressState(BaseModel):
    current_step: int = 0
    total_steps: int = 0
    elapsed_seconds: float = 0.0
    avg_per_step: float = 0.0
    percentage: float = 0.0
    eta_seconds: float = 0.0


class ProgressStore:
    """模拟进度状态管理（单例）"""
    _state: ProgressState

    _instance: ProgressStore | None = None

    def __new__(cls) -> ProgressStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._state: ProgressState = ProgressState()
        self._listeners: list[Callable[[ProgressState], None]] = []

    @property
    def state(self) -> ProgressState:
        return self._state

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        if self._state.total_steps > 0 and self._state.current_step > 0:
            self._state.percentage = min(
                100.0, self._state.current_step / self._state.total_steps * 100
            )
            if self._state.avg_per_step > 0:
                remaining = self._state.total_steps - self._state.current_step
                self._state.eta_seconds = max(0.0, remaining * self._state.avg_per_step)
        self._notify()

    def reset(self):
        self._state = ProgressState()
        self._notify()

    def subscribe(self, callback: Callable[[ProgressState], None]):
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[ProgressState], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self):
        for listener in self._listeners:
            try:
                listener(self._state)
            except Exception:
                import logging
                logging.debug("progress listener failed", exc_info=True)


progress_store = ProgressStore()
