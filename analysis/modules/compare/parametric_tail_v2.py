# analysis/modules/parametric_tail_v2.py

from typing import List, Set, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k as kB, c, m_e, e
from scipy.interpolate import interp1d
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


class ParametricTailV2Module(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "参数扫描 V2：鲁棒非热分析 (中位数锚定法)"

    @property
    def description(self) -> str:
        return "通过中位数锚定热核心温度，利用(均值-中位数)偏离度计算非热份额，并自动扣除初始热底噪。"

    @property
    def required_data(self) -> Set[str]:
        return {'final_spectrum', 'initial_spectrum'}

    # =========================================================================
    # 1. 物理核心：基于中位数的鲁棒温度拟合
    # =========================================================================

    def _get_mj_median_energy(self, T_keV: float) -> float:
        """
        计算给定温度下 Maxwell-Juttner 分布的理论中位数能量 (MeV)。
        由于没有解析解，这里使用数值积分逆运算。
        """
        if T_keV <= 0: return 0.0

        # 定义 theta
        theta = (T_keV * 1e3 * e) / ME_C2_J

        # MJ PDF (unnormalized part) -> f(gamma) ~ gamma * sqrt(gamma^2-1) * exp(-gamma/theta)
        # 换元 E = (gamma - 1) * mc2
        # 我们直接构建一个细密的累积分布函数(CDF)表，然后插值找 0.5 处的值

        # 范围：从 0 到 20 * T (足够覆盖中位数)
        # 能量单位：MeV
        e_grid = np.linspace(0, 20.0 * T_keV * 1e-3, 500)

        # 计算 PDF 值
        # PDF(E) ~ (gamma * beta) * gamma * exp(-gamma/theta)
        # gamma = 1 + E_MeV / 0.511
        mc2_MeV = 0.511
        gamma = 1.0 + e_grid / mc2_MeV
        beta_gamma = np.sqrt(gamma ** 2 - 1.0)
        pdf = beta_gamma * gamma * np.exp(-gamma / theta)

        # 计算 CDF
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]  # 归一化

        # 插值寻找 CDF=0.5 对应的 E
        interp_func = interp1d(cdf, e_grid, kind='linear', bounds_error=False, fill_value="extrapolate")
        return float(interp_func(0.5))

    def _get_mj_mean_energy(self, T_keV: float) -> float:
        """
        计算给定温度下 Maxwell-Juttner 分布的理论平均能量 (MeV)。
        <E> = mc^2 * ( 3*theta + K1(1/th)/K2(1/th) - 1 )
        """
        if T_keV <= 0: return 0.0
        theta = (T_keV * 1e3 * e) / ME_C2_J
        if theta < 1e-9: return 1.5 * kB * (T_keV * 1e3 * e) / J_PER_MEV  # 经典极限

        avg_J = ME_C2_J * (3 * theta + bessel_k(1, 1.0 / theta) / bessel_k(2, 1.0 / theta) - 1.0)
        return avg_J / J_PER_MEV

    def _solve_robust_temperature(self, spec: SpectrumData) -> float:
        """
        核心算法：根据能谱的【中位数能量】反推温度。
        中位数不受高能尾巴影响，能代表 Bulk Temperature。
        """
        if spec.weights.size == 0: return 0.0

        # 1. 计算实验数据的中位数
        # 先对数据排序
        sorted_indices = np.argsort(spec.energies_MeV)
        sorted_e = spec.energies_MeV[sorted_indices]
        sorted_w = spec.weights[sorted_indices]

        cum_w = np.cumsum(sorted_w)
        total_w = cum_w[-1]
        target_w = 0.5 * total_w

        # 找到中位数能量
        idx = np.searchsorted(cum_w, target_w)
        median_e_mev = sorted_e[min(idx, len(sorted_e) - 1)]

        # 2. 求解 T 使得 Theoretical_Median(T) == Experimental_Median
        # 估算初值：对于非相对论，Median ~ 1.386 * T_energy
        # E_median ~ 1.4 * kT -> kT ~ E_median / 1.4
        t_guess_keV = (median_e_mev * 1e3) / 1.4

        try:
            # 建立误差函数
            def err_func(t):
                return self._get_mj_median_energy(t) - median_e_mev

            # 求解
            sol = root_scalar(err_func, x0=t_guess_keV, bracket=[t_guess_keV * 0.1, t_guess_keV * 5.0], method='brentq')
            return sol.root
        except:
            # 如果求解失败，回退到简单的平均值估算（虽然不准，但在极端情况下作为保底）
            return t_guess_keV

    def _calculate_non_thermal_metrics(self, spec: SpectrumData) -> Dict[str, float]:
        """
        计算非热指标。
        """
        if spec is None or spec.weights.size == 0:
            return {'T_core': 0.0, 'excess_ratio': 0.0}

        # 1. 鲁棒拟合核心温度 (基于中位数)
        t_core_keV = self._solve_robust_temperature(spec)

        # 2. 计算该温度下的理论平均能量 (Pure Thermal Mean Energy)
        mean_e_thermal = self._get_mj_mean_energy(t_core_keV)

        # 3. 计算实际平均能量 (Observed Mean Energy)
        total_energy = np.sum(spec.energies_MeV * spec.weights)
        total_weight = np.sum(spec.weights)
        mean_e_obs = total_energy / total_weight

        # 4. 计算非热能量占比
        # 逻辑：总能量 - 热核能量 = 非热注入能量
        # 如果分布完全是热的，Mean ≈ Thermal Mean，结果为 0
        # 如果有尾巴，Mean >> Thermal Mean，结果为正
        excess_energy_per_particle = mean_e_obs - mean_e_thermal

        # 归一化：(E_obs - E_thermal) / E_obs
        # 表示观测到的能量中有百分之多少是不属于那个核心热分布的
        excess_ratio = excess_energy_per_particle / mean_e_obs

        return {
            'T_core': t_core_keV,
            'excess_ratio': excess_ratio,
            'mean_e_obs': mean_e_obs,
            'mean_e_thermal': mean_e_thermal
        }

    # =========================================================================
    # 2. 执行
    # =========================================================================

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 鲁棒非热分析 (Core T Fit & Baseline Subtraction)...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.final_spectrum and r.initial_spectrum]
        if not valid_runs:
            console.print("[red]无有效数据 (需包含 Initial 和 Final spectrum)[/red]")
            return

        # 1. 使用通用工具准备数据 (筛选、排序)
        selector = ParameterSelector(valid_runs)
        x_label, x_vals, sorted_runs = selector.select()

        filename = selector.generate_filename(x_label, sorted_runs, prefix="scan_robust")

        y_excess_net = []  # 最终绘制的净非热占比
        y_temp_core = []  # 核心温度

        console.print(f"{'Run':<40} | {'T_core (keV)':<12} | {'Raw Excess':<12} | {'Base Excess':<12} | {'Net Excess':<12}")
        console.print("-" * 100)

        for run in sorted_runs:
            # 1. 计算初始时刻 (Baseline)
            # 理论上初始时刻是热分布，Excess 应该接近 0。
            # 如果不为 0，说明初始分布与理想 MJ 分布有微小偏差（或网格效应）。
            m_init = self._calculate_non_thermal_metrics(run.initial_spectrum)

            # 2. 计算最终时刻
            m_final = self._calculate_non_thermal_metrics(run.final_spectrum)

            # 3. 差分扣除 (关键步骤)
            # Net Excess = Final_Excess - Initial_Excess
            # 这移除了系统性误差
            net_excess = m_final['excess_ratio'] - m_init['excess_ratio']

            y_excess_net.append(net_excess)
            y_temp_core.append(m_final['T_core'])

            console.print(
                f"{run.name:<40} | {m_final['T_core']:<12.2f} | {m_final['excess_ratio'] * 100:6.3f}%     | {m_init['excess_ratio'] * 100:6.3f}%     | [bold green]{net_excess * 100:6.3f}%[/bold green]")

        # 绘图
        try:
            x_num = [float(v) for v in x_vals]
            is_num = True
        except:
            x_num = range(len(x_vals))
            is_num = False

        with create_analysis_figure(sorted_runs, "scan_robust", num_plots=2, figsize=(9, 8), override_filename=filename) as (fig, (ax1, ax2)):

            # 图 1: 净非热能量占比
            ax1.plot(x_num, np.array(y_excess_net) * 100, 'o-', color='crimson', lw=2, markersize=8)
            ax1.set_ylabel(r"Net Non-thermal Energy Ratio (%)" + "\n" + r"$(E_{obs} - E_{core}) / E_{obs} |_{net}$")
            ax1.set_title(f"Non-thermal Acceleration Efficiency vs {x_label}\n(Robust Core Fit + Baseline Subtraction)", fontsize=13)
            ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # 标注：如果值很小，说明主要是加热
            if max(np.abs(y_excess_net)) < 0.005:  # < 0.5%
                ax1.text(0.5, 0.5, "Dominant Heating Regime\n(No significant tail)",
                         transform=ax1.transAxes, ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            # 图 2: 核心温度 vs 整体平均能量
            # 我们画出 T_core 的变化，这代表“加热”效果
            ax2.plot(x_num, y_temp_core, 's-', color='darkorange', lw=2, label='Core Temperature (Robust)')
            ax2.set_ylabel("Core Temperature $T_{median}$ (keV)")
            ax2.set_xlabel(x_label if is_num else "Simulation Case", fontsize=12)
            ax2.set_title(f"Bulk Heating vs {x_label}", fontsize=13)
            ax2.grid(True, linestyle='--', alpha=0.5)

            if not is_num:
                ax1.set_xticks(x_num)
                ax1.set_xticklabels(x_vals, rotation=45)
                ax2.set_xticks(x_num)
                ax2.set_xticklabels(x_vals, rotation=45)

            plt.subplots_adjust(hspace=0.3)
