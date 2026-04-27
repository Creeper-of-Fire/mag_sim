import asyncio
import os
import subprocess
import sys

from tui.store.app_store import app_store
from tui.store.log_store import logger
from utils.project_config import FILENAME_TASKS_CSV


def open_csv_in_excel():
    """用系统默认程序打开 tasks.csv"""
    if not app_store.job_dir:
        logger.warn("请先选择项目目录。")
        return

    from utils.csv_resolver import resolve_tasks_csv

    csv_path = resolve_tasks_csv(app_store.job_dir)
    if csv_path is None:
        logger.warn(f"未找到 {FILENAME_TASKS_CSV}，请先创建模板。")
        return

    # 跨平台打开
    if sys.platform == "win32":
        os.startfile(csv_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(csv_path)])
    else:
        subprocess.run(["xdg-open", str(csv_path)])

    logger.info(f"已在外部编辑器中打开 {csv_path.name}")
