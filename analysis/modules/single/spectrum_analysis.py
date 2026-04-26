# modules/spectrum_analysis.py

from typing import List

import numpy as np
from scipy.constants import c, m_e, e
from scipy.special import kv

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils import physics_mj
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import AnalysisLayout

# 为了清晰和效率，在模块级别定义常量
ME_C2_J = m_e * c ** 2  # 电子静能量 (单位: 焦耳)
J_PER_MEV = e * 1e6


class SpectrumAnalysisModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "粒子能谱分析"

    @property
    def description(self) -> str:
        return "绘制初始/最终能谱，并与用户输入的理论热谱对比。"

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 粒子能谱分析...[/bold magenta]")

        valid_runs = filter_valid_runs(loaded_runs, require_particles=True, min_particle_files=2)
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
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run)

    def _get_maxwell_juttner(self, E_bins_J: np.ndarray, T_J: float) -> np.ndarray:
        if T_J <= 0: return np.zeros_like(E_bins_J)
        m_e_c2 = m_e * c ** 2
        theta = T_J / m_e_c2
        gamma = 1.0 + E_bins_J / m_e_c2
        pc = np.sqrt(E_bins_J * (E_bins_J + 2 * m_e_c2))
        normalization = 1.0 / (m_e_c2 * theta * kv(2, 1.0 / theta))
        return normalization * (pc / m_e_c2) * gamma * np.exp(-gamma / theta)

    def _generate_single_run_plot(self, run: SimulationRun):
        # 使用布局管理器
        with AnalysisLayout(run, "analysis_spectrum") as layout:
            label_suffix = f" ({run.name})"
            ax = layout.request_axes()

            # 确定能量区间 (bins)
            all_energies = []
            if run.initial_spectrum: all_energies.append(run.initial_spectrum.energies_MeV)
            if run.final_spectrum: all_energies.append(run.final_spectrum.energies_MeV)
            if not all_energies:
                ax.text(0.5, 0.5, '无能谱数据', ha='center', va='center', color='red', transform=ax.transAxes)
                return

            combined = np.concatenate(all_energies)
            positive = combined[combined > 0]
            if positive.size < 2:
                ax.text(0.5, 0.5, '有效能谱数据不足', ha='center', va='center', color='orange', transform=ax.transAxes)
                return

            bins = np.logspace(np.log10(max(positive.min() * 0.5, 1e-4)), np.log10(positive.max() * 1.2), 201)
            centers = np.sqrt(bins[:-1] * bins[1:])
            widths = np.diff(bins)

            # 绘制初始谱
            if run.initial_spectrum:
                counts, _ = np.histogram(run.initial_spectrum.energies_MeV, bins=bins, weights=run.initial_spectrum.weights)
                mask = counts > 0
                ax.plot(centers[mask], (counts / widths)[mask], '--', color='gray', lw=2, label='初始' + label_suffix)

            # 绘制最终谱
            if run.final_spectrum:
                counts, _ = np.histogram(run.final_spectrum.energies_MeV, bins=bins, weights=run.final_spectrum.weights)
                mask = counts > 0
                ax.plot(centers[mask], (counts / widths)[mask], '-', color='royalblue', lw=2.5, label='最终' + label_suffix)

            # 绘制理论热谱
            if run.user_T_keV is not None and run.user_T_keV > 0 and run.initial_spectrum:
                T_J = run.user_T_keV * 1e3 * e
                total_N = np.sum(run.initial_spectrum.weights)
                pdf = self._get_maxwell_juttner(centers * J_PER_MEV, T_J)
                dN_dE = total_N * pdf * J_PER_MEV
                mask = dN_dE > 0
                ax.plot(centers[mask], dN_dE[mask], ':', color='red', lw=2, label=f'理论热谱 (T={run.user_T_keV:.2f} keV)')

            ax.set_title("粒子能谱演化", fontsize=16)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('动能 (MeV)')
            ax.set_ylabel('dN/dE [/MeV]')
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend()
