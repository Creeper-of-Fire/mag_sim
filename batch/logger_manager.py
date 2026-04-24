import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TextIO


class BaseLogger(ABC):
    """日志器抽象基类"""

    @abstractmethod
    def write(self, message: str, end: str = '\n', flush: bool = True):
        """写入日志"""
        pass

    @abstractmethod
    def close(self):
        """关闭日志器"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConsoleLogger(BaseLogger):
    """控制台日志器"""

    def __init__(self, name: str = "console"):
        self.name = name

    def write(self, message: str, end: str = '\n', flush: bool = True):
        print(message, end=end, flush=flush)

    def close(self):
        pass  # 控制台不需要关闭


class FileLogger(BaseLogger):
    """文件日志器"""

    def __init__(self, name: str, file_path: Path):
        self.name = name
        self.file_path = file_path
        self._file_handle: Optional[TextIO] = None

        # 确保目录存在并打开文件
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(file_path, 'w', encoding='utf-8')

    def write(self, message: str, end: str = '\n', flush: bool = True):
        if self._file_handle:
            self._file_handle.write(message + end)
            if flush:
                self._file_handle.flush()

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


class FormattedLogger(BaseLogger):
    """装饰器：为日志添加格式（时间戳、标记等）"""

    def __init__(self, logger: BaseLogger, tag: str = "SYSTEM"):
        self.logger = logger
        self.tag = tag

    def write(self, message: str, end: str = '\n', flush: bool = True):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}][{self.tag}] {message}"
        self.logger.write(formatted, end=end, flush=flush)

    def log_raw(self, message: str, end: str = '', flush: bool = True):
        """直接输出，不添加格式"""
        self.logger.write(message, end=end, flush=flush)

    def log_separator(self, char: str = '=', length: int = 60):
        self.write(char * length)

    def log_task_start(self, task_name: str, task_hash: str):
        self.log_separator()
        self.write(f"开始执行任务: {task_name}")
        self.write(f"任务哈希: {task_hash}")
        self.log_separator()

    def log_task_end(self, task_name: str, status: str, duration_sec: float):
        self.log_separator()
        self.write(f"任务完成: {task_name}")
        self.write(f"最终状态: {status}")
        self.write(f"耗时: {duration_sec:.2f} 秒")
        self.log_separator()

    def close(self):
        self.logger.close()


class LogManager:
    """
    日志管理器
    - 持有一组持久 Logger（通过 add/remove 管理）
    - 支持 with 语句临时添加 Logger（任务级别）
    """

    def __init__(self):
        self._permanent_loggers: list[BaseLogger] = []  # 持久日志器
        self._temporary_loggers: list[BaseLogger] = []  # 临时日志器（with 管理）

    @property
    def _all_loggers(self) -> list[BaseLogger]:
        """所有活跃的日志器"""
        return self._permanent_loggers + self._temporary_loggers

    def add_permanent(self, logger: BaseLogger):
        """添加持久日志器（手动管理生命周期）"""
        self._permanent_loggers.append(logger)

    def remove_permanent(self, logger: BaseLogger):
        """移除持久日志器"""
        if logger in self._permanent_loggers:
            self._permanent_loggers.remove(logger)

    def add_temporary(self, logger: BaseLogger):
        """添加临时日志器（通常由 with 语句自动调用）"""
        self._temporary_loggers.append(logger)

    def remove_temporary(self, logger: BaseLogger):
        """移除临时日志器"""
        if logger in self._temporary_loggers:
            self._temporary_loggers.remove(logger)

    def log(self, message: str, end: str = '\n', flush: bool = True):
        """广播系统消息到所有日志器"""
        for logger in self._all_loggers:
            logger.write(message, end=end, flush=flush)

    def log_raw(self, message: str, end: str = '', flush: bool = True):
        """广播原始消息（无格式化）到所有日志器"""
        for logger in self._all_loggers:
            if isinstance(logger, FormattedLogger):
                logger.log_raw(message, end=end, flush=flush)
            else:
                logger.write(message, end=end, flush=flush)

    def log_system(self, message: str):
        """记录系统日志"""
        self.log(message)

    def log_task_start(self, task_name: str, task_hash: str):
        """记录任务开始"""
        for logger in self._all_loggers:
            if isinstance(logger, FormattedLogger):
                logger.log_task_start(task_name, task_hash)

    def log_task_end(self, task_name: str, status: str, duration_sec: float):
        """记录任务结束"""
        for logger in self._all_loggers:
            if isinstance(logger, FormattedLogger):
                logger.log_task_end(task_name, status, duration_sec)

    def create_task_context(self, task_name: str, log_dir: Path) -> 'TaskLogContext':
        """
        创建任务日志上下文
        用法: with log_manager.create_task_context(task_name, log_dir) as task_ctx:
        """
        return TaskLogContext(self, task_name, log_dir)

    def close_all(self):
        """关闭所有日志器"""
        for logger in self._permanent_loggers + self._temporary_loggers:
            logger.close()
        self._permanent_loggers.clear()
        self._temporary_loggers.clear()


class TaskLogContext:
    """
    任务日志上下文管理器
    进入时自动创建任务日志器并添加，退出时自动移除并关闭
    """

    def __init__(self, log_manager: LogManager, task_name: str, log_dir: Path):
        self.log_manager = log_manager
        self.task_name = task_name
        self.log_dir = log_dir
        self._task_logger: Optional[BaseLogger] = None

    def __enter__(self):
        """进入上下文：创建并添加任务日志器"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        task_log_path = self.log_dir / f"{self.task_name}_{timestamp}.log"

        # 创建文件日志器 + 格式化装饰器
        file_logger = FileLogger(f"task_{self.task_name}", task_log_path)
        self._task_logger = FormattedLogger(file_logger, tag=f"TASK_{self.task_name}")

        # 自动添加到临时列表
        self.log_manager.add_temporary(self._task_logger)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：移除并关闭任务日志器"""
        if self._task_logger:
            self.log_manager.remove_temporary(self._task_logger)
            self._task_logger.close()
            self._task_logger = None


# 便捷工厂函数
def create_standard_log_manager(batch_log_path: Path, tag: str = "BATCH_RUNNER") -> LogManager:
    """
    创建标准的日志管理器
    - 控制台输出（FormattedLogger 包装）
    - 批处理文件输出（FormattedLogger 包装）
    """
    manager = LogManager()

    # 控制台日志器
    console = ConsoleLogger("console")
    formatted_console = FormattedLogger(console, tag=tag)
    manager.add_permanent(formatted_console)

    # 批处理文件日志器
    batch_file = FileLogger("batch_file", batch_log_path)
    formatted_batch = FormattedLogger(batch_file, tag=tag)
    manager.add_permanent(formatted_batch)

    return manager
