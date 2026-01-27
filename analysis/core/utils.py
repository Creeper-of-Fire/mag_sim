# core/utils.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- 通用工具模块 ---
#
# 包含共享的、与具体物理计算无关的辅助函数。
#
import os
import re
from pathlib import Path
from typing import List, Optional, Union

from matplotlib.figure import Figure
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from .config import config

from utils.project_config import PROJECT_ROOT

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


def select_directories() -> List[str]:
    """
    两级选择逻辑：
    1. 选择 Job (位于 sim_jobs/)，并显示每个 Job 下的 Task 数量。
    2. 选择具体的 Task (位于 sim_jobs/<Job>/sim_results/)。
    支持输入格式：单选(1)、多选(1,3)、范围(1-5) 以及混合(1-3, 5, 7-9)。
    """

    # --- 辅助函数：解析范围输入 ---
    def parse_indices(input_str: str, max_len: int) -> List[int]:
        """
        将 "1, 3-5, 8" 这样的字符串解析为 [1, 3, 4, 5, 8]
        并检查是否越界。
        """
        selected_indices = set()
        # 将中文逗号替换为英文逗号，并按逗号或空格分割
        parts = input_str.replace('，', ',').replace(',', ' ').split()

        for part in parts:
            if '-' in part:
                # 处理范围，例如 "1-5"
                try:
                    start_s, end_s = part.split('-')
                    start, end = int(start_s), int(end_s)
                    # 容错：如果用户输入 5-1，自动调整为 1-5
                    if start > end:
                        start, end = end, start
                    # Python range 是左闭右开，所以 end + 1
                    selected_indices.update(range(start, end + 1))
                except ValueError:
                    raise ValueError(f"范围格式无效: {part}")
            else:
                # 处理单个数字
                selected_indices.add(int(part))

        # 转换为列表并排序
        result = sorted(list(selected_indices))

        # 检查越界
        if any(x < 0 or x >= max_len for x in result):
            raise IndexError("索引超出范围")

        return result

    # --- 主逻辑开始 ---
    jobs_root = PROJECT_ROOT / "sim_jobs"

    if not jobs_root.exists():
        console.print(f"[red]错误: 找不到 Jobs 根目录: {jobs_root}[/red]")
        return []

    # --- 第一阶段：选择 Jobs ---
    available_jobs = [d for d in jobs_root.iterdir() if d.is_dir()]
    available_jobs.sort(key=lambda x: natural_sort_key(x.name))

    if not available_jobs:
        console.print("[red]错误: 'sim_jobs' 目录下没有找到任何 Job 文件夹。[/red]")
        return []

    console.print("\n[bold cyan]--- 步骤 1/2: 选择 Job 目录 ---[/bold cyan]")

    # 初始化表格
    job_table = Table(box=box.SIMPLE)
    job_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    job_table.add_column("Job 名称", style="magenta")
    job_table.add_column("包含模拟数", justify="right", style="green")
    job_table.add_column("路径", style="dim")

    # 预扫描并填充数据
    job_candidates = []

    with console.status("[bold green]正在扫描 Job 统计信息...[/bold green]"):
        for i, job_dir in enumerate(available_jobs):
            # 确定搜索范围：优先搜索 sim_results 子目录
            search_scope = job_dir / "sim_results"
            if not search_scope.exists():
                search_scope = job_dir

            # 获取该 Job 下所有的有效运行
            runs_in_job = get_valid_simulation_runs(search_scope)
            count = len(runs_in_job)

            job_candidates.append({
                "dir": job_dir,
                "count": count,
                "runs_cache": runs_in_job
            })

            count_str = str(count) if count > 0 else f"[red]{count}[/red]"

            job_table.add_row(
                str(i),
                job_dir.name,
                count_str,
                str(job_dir.relative_to(PROJECT_ROOT))
            )

    console.print(job_table)

    selected_jobs_indices = []
    while True:
        choice = Prompt.ask(
            "[bold]请输入 Job ID (支持范围如 1-3，回车全选)[/bold]",
            default="all"
        )

        if choice.lower() == 'all':
            # 过滤掉 count=0 的
            selected_jobs_indices = [i for i, c in enumerate(job_candidates) if c['count'] > 0]
            if not selected_jobs_indices:
                console.print("[yellow]警告: 所有 Job 目录似乎都是空的 (Count=0)。[/yellow]")
                return []
            break

        try:
            indices = parse_indices(choice, len(job_candidates))

            # 二次确认：如果选中了空的 Job，给个提示但允许通过，或者自动过滤
            valid_indices = [i for i in indices if job_candidates[i]['count'] > 0]
            if len(valid_indices) < len(indices):
                console.print("[yellow]提示: 已自动忽略部分不包含模拟数据的 Job。[/yellow]")

            if not valid_indices:
                console.print("[red]所选的 Job 均不包含数据，请重新选择。[/red]")
                continue

            selected_jobs_indices = valid_indices
            break

        except ValueError:
            console.print("[red]输入无效，请输入数字或范围(如 0-5)。[/red]")
        except IndexError:
            console.print("[red]索引超出范围，请检查 ID 是否正确。[/red]")

    if not selected_jobs_indices:
        return []

    # --- 第二阶段：收集并选择具体的 Tasks ---
    console.print("\n[bold cyan]--- 步骤 2/2: 选择具体 Simulation Run ---[/bold cyan]")

    all_task_candidates = []

    for idx in selected_jobs_indices:
        job_info = job_candidates[idx]
        job_dir = job_info['dir']
        runs = job_info['runs_cache']

        if not runs:
            continue

        for run_path in runs:
            try:
                rel_name = run_path.relative_to(job_dir / "sim_results")
            except ValueError:
                rel_name = run_path.relative_to(job_dir)

            all_task_candidates.append({
                "path": run_path,
                "job": job_dir.name,
                "name": str(rel_name)
            })

    if not all_task_candidates:
        console.print("[red]所选 Job 中没有包含有效的模拟数据。[/red]")
        return []

    # 展示候选 Run
    run_table = Table(title=f"已选 Job 中的任务列表 (共 {len(all_task_candidates)} 个)", box=box.ROUNDED)
    run_table.add_column("ID", justify="right", style="cyan")
    run_table.add_column("所属 Job", style="blue")
    run_table.add_column("Task/Run 名称", style="green")

    for i, item in enumerate(all_task_candidates):
        run_table.add_row(str(i), item['job'], item['name'])

    console.print(run_table)

    # 选择最终结果
    final_paths = []
    while True:
        choice = Prompt.ask(
            "[bold]请输入 Run ID 进行分析 (支持范围如 10-20，回车全选)[/bold]",
            default="all"
        )

        if choice.lower() == 'all':
            final_paths = [str(c['path']) for c in all_task_candidates]
            break

        try:
            indices = parse_indices(choice, len(all_task_candidates))
            final_paths = [str(all_task_candidates[x]['path']) for x in indices]
            break

        except ValueError:
            console.print("[red]输入无效，请输入数字或范围(如 0-5)。[/red]")
        except IndexError:
            console.print("[red]索引超出范围，请检查 ID 是否正确。[/red]")

    console.print(f"[green]✔ 已选中 {len(final_paths)} 个模拟数据目录。[/green]")
    return final_paths

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
        return run.job_path / config.analysis_folder_name / config.single_analysis_subfolder / run.name

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
