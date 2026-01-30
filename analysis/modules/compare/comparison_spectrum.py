# analysis/modules/comparison_spectrum.py

from typing import List, Set

from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.plotting.layout import create_analysis_figure
from analysis.plotting.spectrum_plotter import SpectrumComparisonPlotter


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
        console.print("\n[bold magenta]执行: 能谱与参数对比分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum and r.final_spectrum.weights.size > 0]

        if len(valid_runs) < 1:  # 即使只有一个run，也允许生成（尽管对比表意义不大）
            console.print(f"[yellow]警告: 找不到有效的能谱数据，无法进行对比分析。[/yellow]")
            return

        if len(valid_runs) < 2:
            console.print(f"[yellow]提示: 只有一个有效的模拟，将只显示其参数表。[/yellow]")

        try:
            # 实例化绘图器，注入所有数据以计算公共分箱
            plotter = SpectrumComparisonPlotter(valid_runs)
        except ValueError as e:
            console.print(f"[red]错误: 初始化绘图器失败: {e}[/red]")
            return

        with create_analysis_figure(valid_runs, "comparison_spectrum", num_plots=1, figsize=(14, 8)) as (fig, ax_plot):
            for run in valid_runs:
                plotter.plot(ax_plot, run, label=run.name)
            plotter.setup_axes(ax_plot)