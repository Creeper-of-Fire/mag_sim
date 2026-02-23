import os
from types import SimpleNamespace
from typing import Set, Optional

import dill

from analysis.core.simulation import SimulationRun
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.core.utils import console


def load_run_data(dir_path: str, required_data: Set[str] = None) -> Optional[SimulationRun]:
    """
    [工厂函数] 为单个模拟目录创建一个 SimulationRun 实例。
    """
    console.print(f"\n[bold cyan]正在初始化模拟: {os.path.basename(dir_path)}[/bold cyan]")

    param_file = os.path.join(dir_path, "sim_parameters.dpkl")
    if not os.path.exists(param_file):
        console.print(f"  [red]✗ 错误: 找不到参数文件 '{param_file}'。[/red]")
        return None

    try:
        with open(param_file, "rb") as f:
            sim_obj = SimpleNamespace(**dill.load(f))

        # 创建实例，它会自动建立索引
        run = SimulationRunSingle(path=dir_path, name=os.path.basename(dir_path), sim=sim_obj)
        console.print("  [green]✔ 索引建立完成。[/green]")
        return run

    except Exception as e:
        console.print(f"  [red]✗ 加载失败: {e}[/red]")
        return None
