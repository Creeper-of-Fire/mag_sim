# tui/config.py - TUI 专用配置（如果需要的话）
"""
TUI 应用的配置常量
"""
from pathlib import Path

# 自动定位项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 直接使用项目已有的配置
from utils.project_config import (
    PROJECT_ROOT,
    FILENAME_HISTORY,
    FILENAME_TASKS_CSV,
    COLUMN_TASK_NAME,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING
)