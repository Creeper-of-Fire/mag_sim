# modules/field_evolution.py

from typing import List, Set

import matplotlib.pyplot as plt

from .base_module import BaseAnalysisModule
from ..core.simulation import SimulationRun
from ..core.utils import console, plot_parameter_table
from ..plotting.field_plotter import FieldRmsPlotter, FieldMeanPlotter, FieldMagnitudePlotter


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

        fig, (ax_rms, ax_mean, ax_mag, ax_table) = plt.subplots(
            4, 1, figsize=(12, 18),
            gridspec_kw={'height_ratios': [3, 3, 3, 3]}, constrained_layout=True
        )
        fig.suptitle(f"磁场演化分析: {run.name}", fontsize=18, y=1.02)

        # --- 使用绘图器 ---
        # 子图1: RMS
        rms_plotter.plot(ax_rms, run, run.name)
        rms_plotter.setup_axes(ax_rms)

        # 子图2: 平均值
        mean_plotter.plot(ax_mean, run, run.name)
        mean_plotter.setup_axes(ax_mean)

        # 子图3: 强度
        mag_plotter.plot(ax_mag, run, run.name)
        mag_plotter.setup_axes(ax_mag)

        # 子图4: 参数表
        plot_parameter_table(ax_table, run)

        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        console.print(f"  [green]✔ 图已保存: {output_name}[/green]")
