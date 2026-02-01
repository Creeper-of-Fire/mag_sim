# core/utils.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 通用工具模块 ---
#
# 包含共享的、与具体物理计算无关的辅助函数。
#
import re
from pathlib import Path
from typing import List, Optional, Union, Dict

from matplotlib.figure import Figure
from rich import box
from rich.console import Console
from rich.table import Table

from utils.project_config import PROJECT_ROOT
from .config import config
from .selector import BaseSelector, SimpleTableSelector

# --- 导入核心数据结构 ---

# --- 全局常量和控制台 ---
console = Console()
from scipy import constants

C = constants.c
M_E = constants.m_e
E = constants.e
J_PER_MEV = E * 1e6


# =============================================================================
# 交互式目录选择
# =============================================================================

def natural_sort_key(s):
    """
    自然排序键函数，用于处理包含数字和字母的字符串排序。
    数字部分按数值大小排序，而非按字符串排序，例如: task1, task2, task10 而不是 task1, task10, task2。
    """
    # 匹配浮点数或整数
    pattern = r'(\d+\.\d+|\d+)'
    parts = re.split(pattern, str(s))

    result = []
    for part in parts:
        if not part:
            continue
        try:
            val = float(part)
            # 数字类型权重设为 0
            # 元组结构: (类型标识, 数值)
            result.append((0, val))
        except ValueError:
            # 字符串类型权重设为 1
            # 元组结构: (类型标识, 字符串值)
            result.append((1, part.lower()))
    return result


def get_valid_simulation_runs(root_path: Path) -> List[Path]:
    """
    递归查找包含 'sim_parameters.dpkl' 的目录。
    现在的结构通常是: JobDir -> sim_results -> TaskDir -> .dpkl
    """
    valid_runs = []
    if not root_path.exists():
        return []

    # 使用 rglob 可以在 sim_results 下查找，适配可能存在的不同层级深度
    for path in root_path.rglob('sim_parameters.dpkl'):
        valid_runs.append(path.parent)

    # 使用 natural_sort_key 排序
    return sorted(valid_runs, key=lambda x: natural_sort_key(x.name))


class JobSelector(BaseSelector[Dict]):
    """
    专门用于选择 Job 的选择器。
    它不仅显示 Job，还预先计算每个 Job 里的 Run 数量。
    """

    def __init__(self, job_dirs: List[Path]):
        # 在 init 中处理数据，将 Path 转换为包含统计信息的字典
        processed_items = []
        with console.status("[bold green]正在扫描 Job 统计信息...[/bold green]"):
            for job_dir in job_dirs:
                search_scope = job_dir / "sim_results"
                if not search_scope.exists(): search_scope = job_dir

                runs = get_valid_simulation_runs(search_scope)
                processed_items.append({
                    "dir": job_dir,
                    "name": job_dir.name,
                    "count": len(runs),
                    "runs": runs  # 缓存起来传给下一步
                })

        super().__init__(processed_items, title="步骤 1/2: 选择 Job 目录")

    def render_menu(self) -> None:
        table = Table(box=box.SIMPLE)
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Job 名称", style="magenta")
        table.add_column("包含模拟数", justify="right", style="green")
        table.add_column("路径", style="dim")

        for i, item in enumerate(self.items):
            count_str = str(item['count']) if item['count'] > 0 else f"[red]{item['count']}[/red]"
            rel_path = str(item['dir'].relative_to(PROJECT_ROOT) if PROJECT_ROOT != Path(".") else item['dir'])
            table.add_row(str(i), item['name'], count_str, rel_path)

        console.print(table)

    # 重写 select 以增加一层业务逻辑：过滤空 Job
    def select(self, default="all", **kwargs):
        # 调用父类的通用选择逻辑
        selected = super().select(default=default, **kwargs)

        # 业务后处理：过滤掉 count == 0 的
        valid_selected = [item for item in selected if item['count'] > 0]

        if len(valid_selected) < len(selected):
            console.print("[yellow]提示: 已自动忽略部分不包含数据的 Job。[/yellow]")

        return valid_selected


class TaskSelector(BaseSelector[Dict]):
    """
    专门用于选择具体 Task (Run) 的选择器。
    """

    def __init__(self, task_dicts: List[Dict]):
        # items 结构: {'path': Path, 'job_name': str, 'name': str}
        super().__init__(task_dicts, title="步骤 2/2: 选择具体 Simulation Run")

    def render_menu(self) -> None:
        table = Table(title=f"任务列表 (共 {len(self.items)} 个)", box=box.ROUNDED)
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("所属 Job", style="blue")
        table.add_column("Task/Run 名称", style="green")

        for i, item in enumerate(self.items):
            table.add_row(str(i), item['job_name'], item['name'])

        console.print(table)


def select_directories() -> List[str]:
    """
    两级选择逻辑：
    1. 选择 Job (位于 sim_jobs/)，并显示每个 Job 下的 Task 数量。
    2. 选择具体的 Task (位于 sim_jobs/<Job>/sim_results/)。
    支持输入格式：单选(1)、多选(1,3)、范围(1-5) 以及混合(1-3, 5, 7-9)。
    """

    # --- 主逻辑开始 ---
    jobs_root = PROJECT_ROOT / "sim_jobs"

    if not jobs_root.exists():
        console.print(f"[red]错误: 找不到 Jobs 根目录: {jobs_root}[/red]")
        return []

    # --- 第一阶段：选择 Jobs ---
    if not jobs_root.exists():
        console.print("[yellow]未找到 standard sim_jobs，切换到简单模式...[/yellow]")
        # 简单模式：直接用 SimpleTableSelector
        current_dirs = [d for d in Path(".").iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not current_dirs: return []

        selector = SimpleTableSelector(
            items=sorted(current_dirs, key=lambda x: x.name),
            columns=["目录名"],
            row_converter=lambda x: [f"[yellow]📂 {x.name}[/yellow]"],
            title="请选择数据目录"
        )
        selected = selector.select()
        return [str(p) for p in selected]

    available_jobs = sorted([d for d in jobs_root.iterdir() if d.is_dir()],
                            key=lambda x: natural_sort_key(x.name))

    if not available_jobs:
        console.print("[red]sim_jobs 下无目录。[/red]")
        return []

    # --- 实例化 JobSelector 并执行选择 ---
    job_selector = JobSelector(available_jobs)
    # 这里的 items 是包含 runs 缓存的字典列表
    selected_job_items = job_selector.select()

    if not selected_job_items:
        return []

    # --- 第二阶段：收集并选择具体的 Tasks ---
    all_tasks = []
    for job_item in selected_job_items:
        job_dir = job_item['dir']
        for run_path in job_item['runs']:
            # 格式化名称
            try:
                if (job_dir / "sim_results").exists():
                    name = str(run_path.relative_to(job_dir / "sim_results"))
                else:
                    name = str(run_path.relative_to(job_dir))
            except ValueError:
                name = run_path.name

            all_tasks.append({
                "path": run_path,
                "job_name": job_item['name'],
                "name": name
            })

    if not all_tasks:
        console.print("[red]所选 Job 中无有效 Task。[/red]")
        return []

        # --- 4. 实例化 TaskSelector 并执行选择 ---
    task_selector = TaskSelector(all_tasks)
    selected_tasks = task_selector.select()

    return [str(t['path']) for t in selected_tasks]


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import SimulationRun


def _determine_output_base_path(run_or_runs: Union['SimulationRun', List['SimulationRun'], None]) -> Path:
    """
    根据传入的模拟对象，智能决定基础输出目录。
    
    逻辑:
    1. None -> 使用全局配置的 output_dir。
    2. 单个 Run -> JobDir/analysis/single_runs。
    3. 多个 Run (属于同一个 Job) -> JobDir/analysis/comparisons。
    4. 多个 Run (跨 Job) -> Global_Dir/cross_job_comparisons。
    """
    if run_or_runs is None:
        return Path(config.global_output_dir)

    # 情况 A: 单个模拟
    if not isinstance(run_or_runs, list):
        run = run_or_runs
        return run.job_path / config.analysis_folder_name / config.single_analysis_subfolder

    # 情况 B: 多个模拟
    runs = run_or_runs
    if not runs:
        return Path(config.global_output_dir)

    # 检查是否所有 run 都来自同一个 Job
    first_job = runs[0].job_path
    all_same_job = all(r.job_path == first_job for r in runs)

    if all_same_job:
        # 同一个 Job 的对比 -> 存入该 Job 的 comparisons 目录
        return first_job / config.analysis_folder_name / config.comparison_subfolder
    else:
        # 跨 Job 对比 -> 存入全局目录
        return Path(config.global_output_dir) / "cross_job_comparisons"


# =============================================================================
# 绘图辅助函数
# =============================================================================
def save_figure(
        fig: Figure,
        filename: str,
        run_or_runs: Union['SimulationRun', List['SimulationRun'], None] = None,
        subfolder: Optional[str] = None
):
    """
    将 Matplotlib Figure 保存到智能计算的目录中。

    Args:
        fig (Figure): 要保存的 Matplotlib Figure 对象。
        filename (str): 输出文件的基本名称。
        run_or_runs: 用于上下文判断的模拟对象（单个或列表）。
                     如果不传，将保存到 config.global_output_dir 根目录。
        subfolder (str, optional): 在智能路径下的进一步子文件夹 (例如 "with_table")。
    """
    # 1. 确定基础路径
    base_dir = _determine_output_base_path(run_or_runs)

    # 2. 拼接子文件夹 (如果有)
    if subfolder:
        output_dir = base_dir / subfolder
    else:
        output_dir = base_dir

    # 3. 确保目录存在
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        console.print(f"[red]创建目录失败 {output_dir}: {e}[/red]")
        return

    output_path = output_dir / filename

    # 4. 保存图像
    # 使用 bbox_inches='tight' 裁剪空白
    try:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')

        # 打印相对路径以便阅读
        try:
            # 尝试相对于项目根目录显示
            display_path = output_path.relative_to(PROJECT_ROOT)
        except ValueError:
            try:
                # 尝试相对于当前目录
                display_path = output_path.relative_to(Path.cwd())
            except ValueError:
                display_path = output_path

        console.print(f"  [green]✔ 图已保存: {display_path}[/green]")

    except Exception as e:
        console.print(f"  [red]✗ 保存图像失败: {e}[/red]")
