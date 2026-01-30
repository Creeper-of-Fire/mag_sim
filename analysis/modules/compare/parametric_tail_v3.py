# analysis/modules/parametric_tail_v3.py

import numpy as np
from typing import List, Set, Dict
from scipy.optimize import root_scalar
from scipy.special import kn as bessel_k
from scipy.constants import k as kB, c, m_e, e

from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.core.parameter_selector import ParameterSelector
from analysis.core.simulation import SimulationRun, SpectrumData
from analysis.core.utils import console
from analysis.plotting.layout import create_analysis_figure

# --- 物理常量 ---
ME_C2_J = m_e * c ** 2
J_PER_MEV = e * 1e6
J_TO_KEV = 1.0 / (e * 1e3)


class ParametricTailV3Module(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "参数扫描 V3：平均能量偏离法 (形状畸变分析)"

    @property
    def description(self) -> str:
        return "利用平均能量锚定等效温度，通过计算模拟能谱相对于理想 MJ 分布的正向面积差来衡量非热成分，并扣除初始噪声。"

    @property
    def required_data(self) -> Set[str]:
        return {'final_spectrum', 'initial_spectrum'}

    # =========================================================================
    # 1. 物理核心算法
    # =========================================================================

    def _solve_temperature_kev(self, avg_ek_mev: float) -> float:
        """根据平均动能反推 Maxwell-Juttner 温度 (keV)"""
        if avg_ek_mev <= 0: return 0.0
        target_avg_ek_j = avg_ek_mev * J_PER_MEV

        def mj_avg_energy_func(T_K):
            if T_K <= 0: return -1.0
            theta = (kB * T_K) / ME_C2_J
            if theta < 1e-9: return 1.5 * kB * T_K
            # <E_k> = mc^2 * ( 3*theta + K1(1/th)/K2(1/th) - 1 )
            return ME_C2_J * (3 * theta + bessel_k(1, 1.0 / theta) / bessel_k(2, 1.0 / theta) - 1.0)

        T_guess = (2.0 / 3.0) * target_avg_ek_j / kB
        try:
            sol = root_scalar(lambda t: mj_avg_energy_func(t) - target_avg_ek_j,
                              x0=T_guess, bracket=[T_guess * 0.1, T_guess * 20.0], method='brentq')
            return (sol.root * e / e) * kB * J_TO_KEV  # 简化单位转换
        except:
            return T_guess * kB * J_TO_KEV

    def _calculate_mj_pdf(self, E_MeV: np.ndarray, T_keV: float) -> np.ndarray:
        """计算 Maxwell-Juttner 概率密度 f(E)"""
        if T_keV <= 0: return np.zeros_like(E_MeV)
        T_J = T_keV * 1e3 * e
        theta = T_J / ME_C2_J

        # 归一化系数 A = 1 / [mc^2 * theta * K2(1/theta)]
        norm = 1.0 / (ME_C2_J * theta * bessel_k(2, 1.0 / theta))

        E_J = E_MeV * J_PER_MEV
        gamma = 1.0 + E_J / ME_C2_J
        pc_J = np.sqrt(E_J * (E_J + 2 * ME_C2_J))

        # PDF (per Joule) -> 转为 per MeV
        pdf = norm * (pc_J / ME_C2_J) * gamma * np.exp(-gamma / theta) * J_PER_MEV
        return pdf

    def _get_spectrum_excess_metrics(self, spec: SpectrumData) -> Dict[str, float]:
        """计算单个能谱的形状偏离度"""
        if spec is None or spec.weights.size == 0:
            return {'T_eff': 0.0, 'excess_ratio': 0.0}

        # 1. 基础统计
        total_weight = np.sum(spec.weights)
        total_energy_MeV = np.sum(spec.energies_MeV * spec.weights)
        avg_energy_MeV = total_energy_MeV / total_weight

        # 2. 拟合等效温度 T_eff
        t_eff_kev = self._solve_temperature_kev(avg_energy_MeV)

        # 3. 建立分箱进行面积对比 (使用 log 分箱覆盖高能尾巴)
        e_min = max(1e-4, spec.energies_MeV.min())
        e_max = max(10.0, spec.energies_MeV.max() * 2.0)
        bins = np.logspace(np.log10(e_min), np.log10(e_max), 300)
        centers = np.sqrt(bins[:-1] * bins[1:])
        widths = np.diff(bins)

        # 模拟数据直方图
        counts_sim, _ = np.histogram(spec.energies_MeV, bins=bins, weights=spec.weights)

        # 理论 MJ 直方图
        pdf_vals = self._calculate_mj_pdf(centers, t_eff_kev)
        counts_th = pdf_vals * widths * total_weight

        # 4. 计算正向能量溢出 (Positive Energy Excess)
        # diff > 0 表示该能量段模拟粒子比理论多（即“尾巴”或“畸变”部分）
        diff_counts = counts_sim - counts_th
        positive_diff = np.maximum(0.0, diff_counts)
        excess_energy_MeV = np.sum(positive_diff * centers)

        return {
            'T_eff': t_eff_kev,
            'excess_ratio': excess_energy_MeV / total_energy_MeV,
            'avg_e': avg_energy_MeV
        }

    # =========================================================================
    # 2. 运行逻辑
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold cyan]执行: 平均能量面积偏离分析 (V3)...[/bold cyan]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum and r.initial_spectrum]
        if not valid_runs: return

        # 参数选择器
        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()
        filename = selector.generate_filename(x_label, sorted_runs, prefix="scan_shape_v3")

        y_net_excess = []
        y_t_eff = []

        console.print(f"{'Run':<40} | {'T_eff(keV)':<12} | {'Raw Exc%':<10} | {'Base Exc%':<10} | {'Net Exc%'}")
        console.print("-" * 100)

        for run in sorted_runs:
            # 计算初始时刻基线
            m_init = self._get_spectrum_excess_metrics(run.initial_spectrum)
            # 计算最终时刻
            m_final = self._get_spectrum_excess_metrics(run.final_spectrum)

            # 净偏离度 = 最终偏离 - 初始偏离 (扣除热噪声和网格效应)
            net_excess = m_final['excess_ratio'] - m_init['excess_ratio']

            y_net_excess.append(net_excess)
            y_t_eff.append(m_final['T_eff'])

            console.print(
                f"{run.name:<40} | {m_final['T_eff']:<12.2f} | "
                f"{m_final['excess_ratio'] * 100:8.4f}% | {m_init['excess_ratio'] * 100:8.4f}% | "
                f"[bold green]{net_excess * 100:8.4f}%[/bold green]"
            )

        # --- 绘图 ---
        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except:
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "scan_shape_v3", num_plots=2, figsize=(9, 8), override_filename=filename) as (fig, (ax1, ax2)):

            # Top: Net Shape Deviation
            ax1.plot(x_num, np.array(y_net_excess) * 100, 'o-', color='royalblue', lw=2, label='Shape Distortion')
            ax1.set_ylabel("Net Shape Excess Ratio (%)")
            ax1.set_title(f"Spectral Distortion (Non-thermal Tail Indicator) vs {x_label}")
            ax1.grid(True, alpha=0.3)
            ax1.axhline(0, color='k', lw=0.8)

            # Bottom: Effective Temperature
            ax2.plot(x_num, y_t_eff, 's-', color='darkred', lw=2, label='Effective T (from <E>)')
            ax2.set_ylabel("Effective Temperature $T_{eff}$ (keV)")
            ax2.set_xlabel(x_label if is_num else "Simulation Index")
            ax2.set_title("Bulk Heating Trend")
            ax2.grid(True, alpha=0.3)

            if not is_num:
                ax1.set_xticks(x_num);
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num);
                ax2.set_xticklabels(x_vals, rotation=45)

        console.print(f"\n[bold green]分析完成。[/bold green] 结果已保存至: {filename}.png")