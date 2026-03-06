# modules/energy_evolution.py

from typing import List

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.energy_plotter import EnergyDensityPlotter, TotalEnergyPlotter
from analysis.plotting.layout import create_analysis_figure


class EnergyEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能量演化分析"

    @property
    def description(self) -> str:
        return "绘制场能（电/磁）、动能和总能量随时间的演化图。"

    def run(self, loaded_runs: List[SimulationRun]):
        """为每个模拟生成能量演化图。"""
        console.print("\n[bold magenta]执行: 能量演化分析...[/bold magenta]")

        valid_runs = filter_valid_runs(
            loaded_runs,
            require_particles=True,
            min_particle_files=2,
            require_fields=True,
            min_field_files=2
        )
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的能量数据，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            output_name = f"{run.name}_analysis_energy_evolution.png"
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run, output_name)

    def _generate_single_run_plot(self, run: SimulationRun, output_name: str):
        # --- 实例化绘图器 ---
        density_plotter = EnergyDensityPlotter()
        total_energy_plotter = TotalEnergyPlotter()

        with create_analysis_figure(run, "analysis_energy_evolution", num_plots=2, figsize=(12, 10)) as (fig, (ax_density, ax_total)):
            density_plotter.plot(ax_density, run, run.name)
            density_plotter.setup_axes(ax_density)

            total_energy_plotter.plot(ax_total, run, run.name)
            total_energy_plotter.setup_axes(ax_total)
