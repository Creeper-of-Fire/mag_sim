# modules/spectrum_analysis.py

from typing import List

import numpy as np
from scipy.constants import c, m_e  # 物理常量

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils import physics_mj
from analysis.plotting.layout import create_analysis_figure
from analysis.plotting.spectrum_plotter import SpectrumPlotter

# 为了清晰和效率，在模块级别定义常量
ME_C2_J = m_e * c ** 2  # 电子静能量 (单位: 焦耳)


class SpectrumAnalysisModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "粒子能谱分析"

    @property
    def description(self) -> str:
        return "绘制初始/最终能谱，并与用户输入的理论热谱对比。"

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 粒子能谱分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.initial_spectrum or r.final_spectrum]
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的能谱数据，跳过此分析。[/yellow]")
            return

        # --- 自动计算等效温度 ---
        console.print("\n" + "=" * 50)
        console.print("[bold yellow]      自动计算等效温度 (能谱分析)[/bold yellow]")
        console.print("=" * 50)

        for run in valid_runs:
            console.print(f"\n[bold]处理模拟: [cyan]{run.name}[/cyan][/bold]")
            if run.final_spectrum and run.final_spectrum.weights.size > 0:
                # 1. 计算加权平均动能
                avg_energy_MeV = np.average(run.final_spectrum.energies_MeV, weights=run.final_spectrum.weights)
                console.print(f"  [green]➔ 计算出的最终加权平均动能: [bold white]{avg_energy_MeV:.6f} MeV[/bold white][/green]")

                # 2. 根据平均动能求解等效温度
                console.print("  [dim]  正在求解麦克斯韦-朱特纳分布的等效温度...[/dim]")
                calculated_T_keV = physics_mj.solve_mj_temperature_kev(avg_energy_MeV)

                if calculated_T_keV is not None:
                    run.user_T_keV = calculated_T_keV
                    console.print(f"  [green]✔ 计算出的等效温度: [bold white]{run.user_T_keV:.3f} keV[/bold white][/green]")
                else:
                    console.print("[yellow]⚠ 温度计算失败，将不绘制理论谱。[/yellow]")
            else:
                console.print("[yellow]⚠ 最终能谱为空，跳过温度计算。[/yellow]")

        console.print("=" * 50 + "\n")

        # --- 循环绘制每个模拟的结果 ---
        for i, run in enumerate(valid_runs):
            output_name = f"{run.name}_analysis_spectrum.png"
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run, output_name)

    def _generate_single_run_plot(self, run: SimulationRun, output_name: str):
        # --- 实例化绘图器 ---
        spectrum_plotter = SpectrumPlotter()

        # 使用布局管理器
        with create_analysis_figure(run, "analysis_spectrum", num_plots=1, figsize=(10, 6)) as (fig, ax_plot):
            # --- 使用绘图器在 ax_plot 上绘图 ---
            spectrum_plotter.plot(ax_plot, run, run.name)
            spectrum_plotter.setup_axes(ax_plot)
