# modules/base_comparison_module.py

from typing import List
from .base_module import BaseAnalysisModule
from core.utils import console

# 导入类型提示，避免循环导入
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.simulation import SimulationRun


class BaseComparisonModule(BaseAnalysisModule):
    """
    对比分析模块的抽象基类。
    它的 run 方法通常会检查是否有足够的数据进行对比，
    然后调用一个绘图函数将所有 runs 的结果绘制在同一张图上。
    """

    def run(self, loaded_runs: List['SimulationRun']):
        """
        执行对比分析的核心方法。
        内置了对所选模拟数量的检查。
        """
        valid_runs = self.pre_run_check(loaded_runs)
        if not valid_runs:
            return

        self.execute_comparison(valid_runs)

    def pre_run_check(self, loaded_runs: List['SimulationRun']) -> List['SimulationRun']:
        """
        在运行前进行检查。子类可以重写此方法以进行更具体的检查。
        """
        # 检查是否至少有两个模拟可供对比
        if len(loaded_runs) < 2:
            console.print(f"[yellow]警告: 模块 '{self.name}' 是一个对比分析，至少需要选择两个模拟才能运行。[/yellow]")
            return []

        # 检查所需数据是否存在
        required_key = list(self.required_data)[0]  # 假设对比模块通常只关心一种数据
        valid_runs = [r for r in loaded_runs if getattr(r, required_key, None)]

        if len(valid_runs) < 2:
            console.print(f"[yellow]警告: 模块 '{self.name}' 至少需要两个包含有效 '{required_key}' 数据的模拟才能运行。[/yellow]")
            return []

        return valid_runs

    def execute_comparison(self, valid_runs: List['SimulationRun']):
        """
        子类需要实现这个方法来执行实际的绘图逻辑。
        """
        # 这是一个占位符，实际的子类应该重写这个方法
        raise NotImplementedError("子类必须实现 'execute_comparison' 方法！")