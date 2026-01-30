# analysis/modules/parametric_tail_debug.py

from typing import List, Set, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k as kB, c, m_e, e
from scipy.optimize import root_scalar
from scipy.special import kn as bessel_k

from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun, SpectrumData
from analysis.core.utils import console
from analysis.plotting.layout import create_analysis_figure

# --- 物理常量 ---
ME_C2_J = m_e * c ** 2
J_PER_MEV = e * 1e6
J_TO_KEV = 1.0 / (e * 1e3)


class ParametricTailDebugModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "DEBUG：非热算法底噪分析"

    @property
    def description(self) -> str:
        return "对比初始时刻(t=0)与最终时刻的'非热能量'计算值，量化算法由统计涨落引起的误差底噪。"

    @property
    def required_data(self) -> Set[str]:
        # 必须同时拥有初始谱和最终谱
        return {'final_spectrum', 'initial_spectrum'}

    # =========================================================================
    # 1. 物理计算核心 (完全复用原算法以复现问题)
    # =========================================================================

    def _solve_temperature_kev(self, avg_ek_mev: float) -> float:
        """根据平均动能反推 Maxwell-Juttner 温度 (keV)"""
        if avg_ek_mev <= 0: return 0.0
        target_avg_ek_j = avg_ek_mev * J_PER_MEV

        def mj_avg_energy(T_K):
            if T_K <= 0: return -1.0
            theta = (kB * T_K) / ME_C2_J
            if theta < 1e-9: return 1.5 * kB * T_K
            return ME_C2_J * (3 * theta + bessel_k(1, 1.0 / theta) / bessel_k(2, 1.0 / theta) - 1.0)

        T_guess = (2.0 / 3.0) * target_avg_ek_j / kB
        try:
            sol = root_scalar(lambda t: mj_avg_energy(t) - target_avg_ek_j,
                              x0=T_guess, bracket=[T_guess * 0.1, T_guess * 10.0], method='brentq')
            return (sol.root * kB) * J_TO_KEV
        except:
            return 0.0

    def _calculate_mj_pdf(self, E_MeV: np.ndarray, T_keV: float) -> np.ndarray:
        """计算 Maxwell-Juttner 概率密度 f(E)"""
        if T_keV <= 0: return np.zeros_like(E_MeV)
        T_J = T_keV * 1e3 * e
        theta = T_J / ME_C2_J
        norm = 1.0 / (ME_C2_J * theta * bessel_k(2, 1.0 / theta))
        E_J = E_MeV * J_PER_MEV
        gamma = 1.0 + E_J / ME_C2_J
        pc_J = np.sqrt(E_J * (E_J + 2 * ME_C2_J))
        pdf = norm * (pc_J / ME_C2_J) * gamma * np.exp(-gamma / theta) * J_PER_MEV
        return pdf

    def _analyze_spectrum_excess(self, spec: SpectrumData) -> Dict[str, float]:
        """
        对任意给定的能谱(SpectrumData)计算正向差值溢出。
        """
        if spec is None or spec.weights.size == 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'total_excess_MeV': 0.0}

        # 1. 基础统计与温度拟合
        total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
        total_weight = np.sum(spec.weights)

        if total_weight == 0:
            return {'T_keV': 0.0, 'excess_ratio': 0.0, 'total_excess_MeV': 0.0}

        avg_energy_MeV = total_energy_MeV / total_weight
        T_keV = self._solve_temperature_kev(avg_energy_MeV)

        # 2. 建立分箱
        min_e = max(1e-4, spec.energies_MeV.min())
        max_e = max(10.0, spec.energies_MeV.max() * 1.5)
        # 使用较细的分箱来捕捉统计涨落
        bins = np.logspace(np.log10(min_e), np.log10(max_e), 200)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)

        # 3. 模拟数据直方图
        counts_sim, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)

        # 4. 理论数据直方图
        pdf_vals = self._calculate_mj_pdf(centers, T_keV)
        counts_th = pdf_vals * widths * total_weight

        # 5. 核心问题所在：计算加权正向差值
        diff_counts = counts_sim - counts_th
        positive_diff = np.maximum(0.0, diff_counts)  # <--- 问题源头：统计涨落被当成了信号

        excess_energy_MeV = np.sum(positive_diff * centers)
        excess_ratio = excess_energy_MeV / total_energy_MeV if total_energy_MeV > 0 else 0.0

        return {
            'T_keV': T_keV,
            'excess_ratio': excess_ratio,
            'total_excess_MeV': excess_energy_MeV,
            'total_energy_MeV': total_energy_MeV
        }


    # =========================================================================
    # 2. 运行与绘图
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 算法底噪分析 (T=0 vs T=End)...[/bold magenta]")

        # 过滤掉没有初始谱的数据
        valid_runs = []
        for r in loaded_runs:
            if r.final_spectrum and r.initial_spectrum:
                valid_runs.append(r)
            else:
                console.print(f"[yellow]警告: 模拟 {r.name} 缺少 initial_spectrum 或 final_spectrum，已跳过。[/yellow]")

        if len(valid_runs) < 1:
            console.print("[red]错误: 没有足够的数据进行对比。请确保模拟运行时保存了 Step 0 数据。[/red]")
            return

        # 1. 使用 Selector
        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()

        # 2. 生成文件名
        final_filename = selector.generate_filename(x_label, sorted_runs, prefix="debug_tail")

        # 数据容器
        y_ratio_init = []
        y_ratio_final = []
        y_temp_init = []
        y_temp_final = []

        console.print(f"  正在计算 Initial (底噪) 与 Final (信号) ...")

        for i, run in enumerate(sorted_runs):
            m_init = self._analyze_spectrum_excess(run.initial_spectrum)
            m_final = self._analyze_spectrum_excess(run.final_spectrum)

            y_ratio_init.append(m_init['excess_ratio'])
            y_ratio_final.append(m_final['excess_ratio'])
            y_temp_init.append(m_init['T_keV'])
            y_temp_final.append(m_final['T_keV'])

            console.print(f"    [{run.name}] {x_label}={x_vals[i]}")
            console.print(f"      Initial(T=0):  Excess={m_init['excess_ratio'] * 100:6.3f}% (Noise), T={m_init['T_keV']:.2f} keV")
            console.print(f"      Final  (T=end): Excess={m_final['excess_ratio'] * 100:6.3f}% (Signal), T={m_final['T_keV']:.2f} keV")

        # 绘图
        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except:
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "debug_tail", num_plots=2, figsize=(9, 8), override_filename=final_filename) as (fig, (ax1, ax2)):

            # --- 图1: 信号 vs 底噪 ---
            ax1.plot(x_num, np.array(y_ratio_final) * 100, 'o-', color='crimson', lw=2, label='Final Spectrum (Signal + Noise)')
            ax1.plot(x_num, np.array(y_ratio_init) * 100, 'o--', color='gray', lw=2, alpha=0.7, label='Initial Spectrum (Algorithm Noise)')

            # 填充差值区域
            ax1.fill_between(x_num, np.array(y_ratio_init) * 100, np.array(y_ratio_final) * 100,
                             color='crimson', alpha=0.1, label='Net Non-thermal Estimate')

            ax1.set_ylabel("Calculated Non-Thermal Ratio (%)")
            ax1.set_title(f"Check for Algorithm Artifacts: Excess Energy vs {x_label}", fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # --- 图2: 温度对比 ---
            ax2.plot(x_num, y_temp_final, 's-', color='darkorange', lw=2, label='Final T')
            ax2.plot(x_num, y_temp_init, 's--', color='steelblue', lw=2, alpha=0.7, label='Initial T')
            ax2.set_ylabel("Fitted Temperature (keV)")
            ax2.set_xlabel(x_label if is_num else "Simulation Case", fontsize=12)
            ax2.set_title(f"Temperature Heating: {x_label}", fontsize=14)
            ax2.legend(fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.5)

            if not is_num:
                ax1.set_xticks(x_num)
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45)

            plt.subplots_adjust(hspace=0.3)

            console.print("\n[bold green]分析完成。[/bold green]")
            console.print("如果 'Initial Spectrum' 的灰色虚线很高(例如 > 1%)，说明当前的正向差值算法(Positive Excess)受统计涨落影响严重。")
            console.print("建议：改用 Quantile (分位数) 分析或更高阶的拟合方法。")