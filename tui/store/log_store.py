"""
日志中心 - 单例模式，全局日志状态管理
类似 Pinia store，任何模块都可以写入日志
"""
from __future__ import annotations
from collections import deque
from datetime import datetime
from typing import Callable


class LogStore:
    """全局日志存储（单例）"""

    _instance: LogStore | None = None

    def __new__(cls) -> LogStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._logs: list[str] = []
        self._listeners: list[Callable[[str], None]] = []

    # ── 写入日志 ──

    def write(self, message: str):
        """写入一条日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self._logs.append(line)
        self._notify(line)

    def info(self, message: str):
        self.write(f"ℹ️ {message}")

    def warn(self, message: str):
        self.write(f"⚠️ {message}")

    def error(self, message: str):
        self.write(f"❌ {message}")

    # ── 读取日志 ──

    @property
    def all(self) -> list[str]:
        """获取所有日志"""
        return list(self._logs)

    @property
    def last(self) -> str:
        """获取最后一条日志"""
        return self._logs[-1] if self._logs else ""

    def clear(self):
        """清空日志"""
        self._logs.clear()

    # ── 订阅 ──

    def subscribe(self, callback: Callable[[str], None]):
        """订阅日志更新（用于 LogPanel 等 UI 组件）"""
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[str], None]):
        """取消订阅"""
        if callback in self._listeners:
            self._listeners.remove(callback)

    # ── 内部 ──

    def _notify(self, line: str):
        """通知所有监听者"""
        for listener in self._listeners:
            try:
                listener(line)
            except Exception:
                pass


# 全局单例
logger = LogStore()