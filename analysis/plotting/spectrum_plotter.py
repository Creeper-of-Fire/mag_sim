# plotting/spectrum_plotter.py
from typing import Optional, List

import numpy as np
from matplotlib.axes import Axes
from scipy.constants import c, m_e, e
from scipy.special import kv

from .base_plotter import BasePlotter
from ..core.simulation import SimulationRun
from ..modules.utils.comparison_utils import create_common_energy_bins

J_PER_MEV = e * 1e6


class SpectrumPlotter(BasePlotter):
    """绘制粒子能谱。"""

    def _get_maxwell_juttner(self, E_bins_J: np.ndarray, T_J: float) -> np.ndarray:
        if T_J <= 0: return np.zeros_like(E_bins_J)
        m_e_c2 = m_e * c ** 2
        theta = T_J / m_e_c2
        gamma = 1.0 + E_bins_J / m_e_c2
        pc = np.sqrt(E_bins_J * (E_bins_J + 2 * m_e_c2))
        normalization = 1.0 / (m_e_c2 * theta * kv(2, 1.0 / theta))
        return normalization * (pc / m_e_c2) * gamma * np.exp(-gamma / theta)

    def plot(self, ax: Axes, run: SimulationRun, label: str, color: Optional[str] = None, **kwargs):
        label_suffix = f" ({label})"

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

    def setup_axes(self, ax: Axes):
        ax.set_title("粒子能谱演化", fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('动能 (MeV)')
        ax.set_ylabel('dN/dE [/MeV]')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()


class SpectrumComparisonPlotter(BasePlotter):
    """
    绘制多个模拟的能谱对比图。
    它在构造时接收所有模拟数据以预计算统一的能量分箱，
    但其 plot 方法一次只绘制一个模拟的数据。
    """

    def __init__(self, runs: List[SimulationRun]):
        """
        构造函数，负责数据预处理和统一分箱。
        Args:
            runs (List[SimulationRun]): 包含所有待对比模拟的列表。
        """
        self.runs = runs
        self._initial_plotted_label = False  # 用于确保初始谱图例只显示一次

        # --- 调用共享工具函数 ---
        # 这里直接在构造函数中调用，并将结果存为实例变量
        try:
            self.bins, self.centers, self.widths = create_common_energy_bins(runs, num_bins=201)
        except ValueError as e:
            # 在绘图器中，如果分箱失败，我们可以设置一个标志位，在 plot 时提示
            print(f"警告：SpectrumComparisonPlotter 创建分箱失败: {e}")
            self.bins, self.centers, self.widths = None, None, None

    def plot(self, ax: Axes, run: SimulationRun, label: str, color: Optional[str] = None, **kwargs):
        """
        在给定的 Axes 上绘制单个模拟的初始和最终能谱。
        """
        run_color = color

        # 1. 绘制最终能谱 (实线)
        if run.final_spectrum and run.final_spectrum.weights.size > 0:
            counts_final, _ = np.histogram(
                run.final_spectrum.energies_MeV,
                bins=self.bins,
                weights=run.final_spectrum.weights
            )
            normalized_counts = counts_final / self.widths
            mask = normalized_counts > 0

            # 绘制并捕获颜色，以便初始谱复用
            lines = ax.plot(
                self.centers[mask],
                normalized_counts[mask],
                '-',
                lw=2.5,
                label=f"{label} (最终)",
                color=run_color,  # 使用传入的颜色或 matplotlib 自动选择
                **kwargs
            )
            # 如果颜色是自动选择的，就保存下来
            if run_color is None:
                run_color = lines[0].get_color()

        # 2. 绘制初始能谱 (虚线)，使用与最终谱相同的颜色
        if run.initial_spectrum and run.initial_spectrum.weights.size > 0:
            counts_initial, _ = np.histogram(
                run.initial_spectrum.energies_MeV,
                bins=self.bins,
                weights=run.initial_spectrum.weights
            )
            normalized_counts = counts_initial / self.widths
            mask = normalized_counts > 0
            ax.plot(
                self.centers[mask],
                normalized_counts[mask],
                '--',
                lw=1.5,
                label=f"{label} (初始)",
                color=run_color,  # 强制使用相同颜色
                **kwargs
            )

    def setup_axes(self, ax: Axes):
        """配置对比图的坐标轴。"""
        ax.set_title("初始与最终粒子能谱对比", fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('动能 (MeV)')
        ax.set_ylabel('dN/dE [/MeV]')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(title="模拟", fontsize=10, ncol=2)  # 使用多列图例以防太长
