# plotting/spectrum_plotter.py
from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from scipy.constants import c, m_e, e
from scipy.special import kv

from .base_plotter import BasePlotter
from ..core.simulation import SimulationRun

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
