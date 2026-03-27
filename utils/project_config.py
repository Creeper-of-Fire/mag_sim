# project_config.py

from pathlib import Path

# --- 基础路径定义 ---
# 自动定位项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 文件名常量 (统一管理，防止拼写错误)
FILENAME_QUEUE = 'queue.jsonl'
FILENAME_HISTORY = 'history.jsonl'
FILENAME_TASKS_CSV = 'tasks.csv'
FILENAME_DEFAULT_PARAMS = 'default_params.json'
DIRNAME_LOGS = 'logs'

# GUI 显示用的列名
COLUMN_TASK_NAME = '任务名'

# 任务状态常量
STATUS_COMPLETED = "已完成"
STATUS_FAILED = "失败"
STATUS_PENDING = "待运行"
STATUS_RUNNING = "运行中"  # 预留