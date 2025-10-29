# modules/energy_evolution.py

from typing import List, Set

import matplotlib.pyplot as plt

from .base_module import BaseAnalysisModule
from ..core.simulation import SimulationRun
from ..core.utils import console, plot_parameter_table, save_figure
from ..plotting.energy_plotter import EnergyDensityPlotter, TotalEnergyPlotter


class EnergyEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能量演化分析"

    @property
    def description(self) -> str:
        return "绘制场能（电/磁）、动能和总能量随时间的演化图。"

    @property
    def required_data(self) -> Set[str]:
        return {'energy', 'initial_spectrum'}

    def run(self, loaded_runs: List[SimulationRun]):
        """为每个模拟生成能量演化图。"""
        console.print("\n[bold magenta]执行: 能量演化分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.energy_data]
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的能量数据，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            output_name = f"analysis_energy_evolution_{run.name}.png"
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run, output_name)

    def _generate_single_run_plot(self, run: SimulationRun, output_name: str):
        # --- 实例化绘图器 ---
        density_plotter = EnergyDensityPlotter()
        total_energy_plotter = TotalEnergyPlotter()

        fig, (ax_density, ax_total, ax_table) = plt.subplots(
            3, 1, figsize=(12, 18),
            gridspec_kw={'height_ratios': [4, 4, 3]}, constrained_layout=True
        )
        fig.suptitle(f"能量演化分析: {run.name}", fontsize=18, y=1.02)

        # --- 使用绘图器 ---
        # 子图1: 平均能量密度
        density_plotter.plot(ax_density, run, run.name)
        density_plotter.setup_axes(ax_density)

        # 子图2: 总能量
        total_energy_plotter.plot(ax_total, run, run.name)
        total_energy_plotter.setup_axes(ax_total)

        # 子图3: 参数表 (保持不变)
        plot_parameter_table(ax_table, run)

        save_figure(fig, output_name)
