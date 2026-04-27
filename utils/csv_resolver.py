"""
CSV 路径解析器
统一管理 tasks.csv 的定位、重命名、创建、清理
"""
from pathlib import Path
from typing import Optional

from utils.project_config import FILENAME_TASKS_CSV


def resolve_tasks_csv(job_dir: Path) -> Optional[Path]:
    """
    按优先级解析 tasks.csv 的实际路径：

    1. {job_dir.name}_tasks.csv
    2. tasks.csv → 自动重命名为 {job_dir.name}_tasks.csv
    3. *_tasks.csv（仅1个）→ 自动重命名为 {job_dir.name}_tasks.csv
    4. 多个 *_tasks.csv → 选最新的，重命名为 {job_dir.name}_tasks.csv
    5. 没有任何 CSV → 返回 None

    Args:
        job_dir: 项目目录

    Returns:
        解析后的 CSV 路径，如果目录不存在，返回 None。
    """
    preferred = job_dir / f"{job_dir.name}_{FILENAME_TASKS_CSV}"

    # 优先级 1：带项目名的已经存在
    if preferred.exists():
        return preferred

    # 优先级 2：默认名存在 → 重命名
    default = job_dir / FILENAME_TASKS_CSV
    if default.exists():
        default.rename(preferred)
        return preferred

    # 3-4. *_tasks.csv 模式
    wildcard = sorted(job_dir.glob(f"*_{FILENAME_TASKS_CSV}"))
    if len(wildcard) == 1:
        wildcard[0].rename(preferred)
        return preferred
    elif len(wildcard) > 1:
        # 多个同名模式：选最新的
        newest = max(wildcard, key=lambda p: p.stat().st_mtime)
        newest.rename(preferred)
        return preferred

    # 5. 没有任何 CSV
    return None


def get_preferred_csv_path(job_dir: Path) -> Path:
    """
    获取期望的 CSV 路径（不检查是否存在）。
    用于生成模板时指定输出路径。
    """
    return job_dir / f"{job_dir.name}_{FILENAME_TASKS_CSV}"