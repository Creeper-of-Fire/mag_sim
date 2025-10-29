# analysis/modules/comparison_spectrum.py

from typing import List, Set

import matplotlib.pyplot as plt

from .base_module import BaseComparisonModule
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..plotting.spectrum_plotter import SpectrumComparisonPlotter


class SpectrumComparisonModule(BaseComparisonModule):
    """
    对比分析模块：对比不同模拟的最终粒子能谱。
    """
    @property
    def name(self) -> str:
        return "能谱对比分析"

    @property
    def description(self) -> str:
        return "将多个模拟的最终能谱绘制在同一张图上进行对比。"

    @property
    def required_data(self) -> Set[str]:
        return {'initial_spectrum', 'final_spectrum'}

    def run(self, loaded_runs: List[SimulationRun]):
        """
        执行对比分析和绘图。
        模块负责创建画布并遍历模拟，调用绘图器进行绘制。
        """
        console.print("\n[bold magenta]执行: 能谱对比分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum and r.final_spectrum.weights.size > 0]

        if len(valid_runs) < 2:
            console.print(f"[yellow]警告: 找到的有效能谱数据不足两个 ({len(valid_runs)}个)，无法进行对比分析。[/yellow]")
            return

        output_name = "comparison_spectrum_final.png"
        console.print(f"  -> 正在为 {len(valid_runs)} 个模拟生成对比图...")

        try:
            # 1. 实例化绘图器，注入所有数据以计算公共分箱
            plotter = SpectrumComparisonPlotter(valid_runs)
        except ValueError as e:
            console.print(f"[red]错误: 初始化绘图器失败: {e}[/red]")
            return

        # 2. 创建画布
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle("最终能谱对比分析", fontsize=18)

        # 3. 模块负责遍历，在同一个 ax 上绘制每个模拟的谱线
        for run in valid_runs:
            plotter.plot(ax, run, label=run.name)

        # 4. 设置坐标轴样式
        plotter.setup_axes(ax)

        # 5. 保存图像
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        console.print(f"  [green]✔ 对比图已保存: {output_name}[/green]")