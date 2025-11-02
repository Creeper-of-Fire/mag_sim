# modules/field_evolution.py

from typing import List, Set

import matplotlib.pyplot as plt

from .base_module import BaseAnalysisModule
from ..core.simulation import SimulationRun
from ..core.utils import console, plot_parameter_table, save_figure
from ..plotting.field_plotter import FieldRmsPlotter, FieldMeanPlotter, FieldMagnitudePlotter
from ..plotting.layout import create_analysis_figure


class FieldEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "磁场演化分析"

    @property
    def description(self) -> str:
        return "绘制磁场分量的RMS/平均值以及总场强随时间的演化。"

    @property
    def required_data(self) -> Set[str]:
        return {'field', 'initial_spectrum'}

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 磁场演化分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.field_data]
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的场数据，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            output_name = f"analysis_field_evolution_{run.name}.png"
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run, output_name)

    def _generate_single_run_plot(self, run: SimulationRun, output_name: str):
        # --- 实例化绘图器 ---
        rms_plotter = FieldRmsPlotter()
        mean_plotter = FieldMeanPlotter()
        mag_plotter = FieldMagnitudePlotter()

        with create_analysis_figure(run, "analysis_field_evolution", num_plots=3, figsize=(12, 15)) as (fig, (ax_rms, ax_mean, ax_mag)):
            rms_plotter.plot(ax_rms, run, run.name)
            rms_plotter.setup_axes(ax_rms)

            mean_plotter.plot(ax_mean, run, run.name)
            mean_plotter.setup_axes(ax_mean)

            mag_plotter.plot(ax_mag, run, run.name)
            mag_plotter.setup_axes(ax_mag)
