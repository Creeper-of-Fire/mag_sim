# modules/spectrum_analysis.py

from typing import List, Set

import matplotlib.pyplot as plt
import numpy as np
from rich.prompt import Prompt

from .base_module import BaseAnalysisModule
from ..core.simulation import SimulationRun
from ..core.utils import console, plot_parameter_table
from ..plotting.spectrum_plotter import SpectrumPlotter


class SpectrumAnalysisModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "粒子能谱分析"

    @property
    def description(self) -> str:
        return "绘制初始/最终能谱，并与用户输入的理论热谱对比。"

    @property
    def required_data(self) -> Set[str]:
        return {'initial_spectrum', 'final_spectrum'}

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 粒子能谱分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.initial_spectrum or r.final_spectrum]
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的能谱数据，跳过此分析。[/yellow]")
            return

        self._interactive_temperature_input(valid_runs)

        for i, run in enumerate(valid_runs):
            output_name = f"analysis_spectrum_{run.name}.png"
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run, output_name)

    def _interactive_temperature_input(self, runs: List[SimulationRun]):
        console.print("\n" + "=" * 50)
        console.print("[bold yellow]      交互式温度输入环节 (能谱分析)[/bold yellow]")
        console.print("=" * 50)
        for run in runs:
            console.print(f"\n[bold]处理模拟: [cyan]{run.name}[/cyan][/bold]")
            if not run.final_spectrum or run.final_spectrum.weights.size == 0:
                console.print("[yellow]⚠ 最终能谱为空，跳过温度输入。[/yellow]")
                continue
            avg_energy_MeV = np.average(run.final_spectrum.energies_MeV, weights=run.final_spectrum.weights)
            console.print(f"  [green]➔ 计算出的最终加权平均动能为: [bold white]{avg_energy_MeV:.6f} MeV[/bold white][/green]")
            try:
                user_temp = Prompt.ask(f"  [bold spring_green2]请输入为此模拟计算出的温度 (keV) (留空则跳过)[/bold spring_green2]", default="")
                if user_temp:
                    run.user_T_keV = float(user_temp)
                    console.print(f"  [green]✔ 已记录温度: {run.user_T_keV:.2f} keV[/green]")
            except (ValueError, TypeError):
                console.print("[yellow]⚠ 输入无效，将不绘制理论谱。[/yellow]")

    def _generate_single_run_plot(self, run: SimulationRun, output_name: str):
        # --- 实例化绘图器 ---
        spectrum_plotter = SpectrumPlotter()

        fig, (ax_plot, ax_table) = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [3, 2]})
        fig.suptitle(f"能谱分析: {run.name}", fontsize=20, y=0.98)

        # --- 使用绘图器 ---
        spectrum_plotter.plot(ax_plot, run)
        spectrum_plotter.setup_axes(ax_plot)

        # --- 参数表 ---
        plot_parameter_table(ax_table, run)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        console.print(f"  [green]✔ 图已保存: {output_name}[/green]")
