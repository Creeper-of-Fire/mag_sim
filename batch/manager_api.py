# batch/manager_api.py
from enum import Enum, auto
from abc import ABC, abstractmethod

class JobStatus(Enum):
    PENDING = auto()   # 提交中/准备中
    RUNNING = auto()   # 正在运行
    SUCCESS = auto()   # 成功完成
    FAILED = auto()    # 执行失败（模拟器崩溃或环境错误）
    CANCELLED = auto() # 被手动中断
    UNKNOWN = auto()   # 失去连接或状态异常

class BaseComputeManager(ABC):
    """
    计算任务管理器基类。
    一个 Manager 实例负责 一次 任务的提交、监控、日志获取。
    """
    @abstractmethod
    def submit(self, task_hash: str, params: dict, output_dir_name: str):
        """异步提交任务到计算环境。"""
        pass

    @abstractmethod
    def get_status(self) -> JobStatus:
        """获取当前任务的抽象状态。"""
        pass

    @abstractmethod
    def get_logs(self) -> list[str]:
        """获取自上次调用以来的增量日志。"""
        pass

    @abstractmethod
    def interrupt(self):
        """强制中断当前任务。"""
        pass