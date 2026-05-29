# analysis/core/group_merger.py
"""
独立的分组合并工具，供 single 模块按需调用。

从 ParameterSelector 中提取的统计性重复检测与合并逻辑，
不涉及参数选择 / X 轴分析，仅做分组检测 + 交互式确认 + SimulationRunGroup 构建。
"""

import json
from typing import List

from rich.prompt import Confirm

from .simulation import SimulationRun
from .simulationGroup import SimulationRunGroup
from .simulationSingle import SimulationRunSingle
from .utils import console, get_run_parameters


def detect_and_merge_groups(
        runs: List[SimulationRun],
        *,
        interactive: bool = True,
) -> List[SimulationRun]:
    """
    检测统计性重复模拟（参数相同、仅 run_id 不同），并合并为 SimulationRunGroup。

    Args:
        runs: 原始 SimulationRun 列表。
        interactive: 是否交互式确认合并；False 时静默合并。

    Returns:
        合并后的列表——统计性重复被 SimulationRunGroup 替换，其余 run 保持不变。
        已有的 SimulationRunGroup 直接透传，防止双重合并。
    """
    if not runs:
        return runs

    # 分离已有的 Group 和待分组的 Single
    existing_groups: List[SimulationRun] = []
    singles: List[SimulationRunSingle] = []
    for r in runs:
        if isinstance(r, SimulationRunGroup):
            existing_groups.append(r)
        elif isinstance(r, SimulationRunSingle):
            singles.append(r)
        else:
            existing_groups.append(r)  # 未知类型，直接透传

    if not singles:
        return runs

    # 加载参数并按参数分组（排除 run_id）
    groups: dict[str, list[SimulationRunSingle]] = {}
    for r in singles:
        params = get_run_parameters(r)
        params.pop('run_id', None)
        key = json.dumps(params, sort_keys=True)
        groups.setdefault(key, []).append(r)

    # 筛选出统计性重复组
    statistical_groups = {k: v for k, v in groups.items() if len(v) > 1}

    if not statistical_groups:
        return runs  # 无重复，原样返回

    # 交互式确认
    if interactive:
        console.print(
            f"\n[bold yellow]检测到 {len(statistical_groups)} 组统计性重复模拟。[/bold yellow]"
        )
        if not Confirm.ask("是否要将这些重复模拟的结果进行平均（推荐）?", default=True):
            return runs  # 用户拒绝，保持原样

    # 构建合并后的列表
    merged: List[SimulationRun] = existing_groups.copy()
    for k, v in groups.items():
        if len(v) == 1:
            merged.append(v[0])
        else:
            merged.append(SimulationRunGroup(v))

    console.print(
        f"[green]✔ 已合并。当前处理 {len(merged)} 个独立实体。[/green]"
    )
    return merged
