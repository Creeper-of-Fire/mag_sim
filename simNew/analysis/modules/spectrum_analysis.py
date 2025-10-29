# modules/spectrum_analysis.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv
from rich.prompt import Prompt
from typing import List, Set

from .base_module import BaseAnalysisModule
from core.simulation import SimulationRun
from core.utils import console, plot_parameter_table, C, M_E, E, J_PER_MEV


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
            self._generate_plot(run, output_name)

    def _get_maxwell_juttner(self, E_bins_J: np.ndarray, T_J: float) -> np.ndarray:
        if T_J <= 0: return np.zeros_like(E_bins_J)
        m_e_c2 = M_E * C ** 2
        theta = T_J / m_e_c2
        gamma = 1.0 + E_bins_J / m_e_c2
        pc = np.sqrt(E_bins_J * (E_bins_J + 2 * m_e_c2))
        normalization = 1.0 / (m_e_c2 * theta * kv(2, 1.0 / theta))
        return normalization * (pc / m_e_c2) * gamma * np.exp(-gamma / theta)

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

    def _generate_plot(self, run: SimulationRun, output_name: str):
        fig, (ax_plot, ax_table) = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [3, 2]})
        fig.suptitle(f"能谱分析: {run.name}", fontsize=20, y=0.98)

        ax_plot.set_title("粒子能谱演化", fontsize=16)
        all_energies = []
        if run.initial_spectrum: all_energies.append(run.initial_spectrum.energies_MeV)
        if run.final_spectrum: all_energies.append(run.final_spectrum.energies_MeV)

        if not all_energies:
            ax_plot.text(0.5, 0.5, '无能谱数据', ha='center', va='center', color='red')
        else:
            combined = np.concatenate(all_energies)
            positive = combined[combined > 0]
            if positive.size > 1:
                bins = np.logspace(np.log10(max(positive.min() * 0.5, 1e-4)), np.log10(positive.max() * 1.2), 201)
                centers = np.sqrt(bins[:-1] * bins[1:])
                widths = np.diff(bins)

                if run.initial_spectrum:
                    counts, _ = np.histogram(run.initial_spectrum.energies_MeV, bins=bins, weights=run.initial_spectrum.weights)
                    mask = counts > 0
                    ax_plot.plot(centers[mask], (counts / widths)[mask], '--', color='gray', lw=2, label='初始')

                if run.final_spectrum:
                    counts, _ = np.histogram(run.final_spectrum.energies_MeV, bins=bins, weights=run.final_spectrum.weights)
                    mask = counts > 0
                    ax_plot.plot(centers[mask], (counts / widths)[mask], '-', color='royalblue', lw=2.5, label='最终')

                if run.user_T_keV is not None and run.user_T_keV > 0 and run.initial_spectrum:
                    T_J = run.user_T_keV * 1e3 * E
                    total_N = np.sum(run.initial_spectrum.weights)
                    pdf = self._get_maxwell_juttner(centers * J_PER_MEV, T_J)
                    dN_dE = total_N * pdf * J_PER_MEV
                    mask = dN_dE > 0
                    ax_plot.plot(centers[mask], dN_dE[mask], ':', color='red', lw=2, label=f'理论热谱 (T={run.user_T_keV:.2f} keV)')

            ax_plot.set_xscale('log');
            ax_plot.set_yscale('log')
            ax_plot.set_xlabel('动能 (MeV)');
            ax_plot.set_ylabel('dN/dE [/MeV]')
            ax_plot.grid(True, which="both", ls="--", alpha=0.5);
            ax_plot.legend()

        plot_parameter_table(ax_table, run)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        console.print(f"  [green]✔ 图已保存: {output_name}[/green]")