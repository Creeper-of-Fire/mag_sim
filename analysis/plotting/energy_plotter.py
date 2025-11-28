# plotting/energy_plotter.py
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.constants import e

from .base_plotter import BasePlotter
from ..core.simulation import SimulationRun


class EnergyDensityPlotter(BasePlotter):
    """绘制平均能量密度随时间的演化。"""

    def plot(self, ax: Axes, run: SimulationRun, label: str, color: Optional[str] = None, **kwargs):
        data = run.energy_data
        J_PER_EV = e

        label_suffix = f" ({label})" if label else ""

        # 动能
        ax.plot(data.time, data.mean_kin_energy_density / J_PER_EV,
                label=r'$\langle \epsilon_K \rangle$' + label_suffix, **kwargs)

        # 磁能
        ax.plot(data.time, data.mean_mag_energy_density_total / J_PER_EV, '--',
                label=r'$\langle \epsilon_B \rangle$' + label_suffix, **kwargs)

        # 电能
        ax.plot(data.time, data.mean_elec_energy_density_total / J_PER_EV, '--',
                label=r'$\langle \epsilon_E \rangle$' + label_suffix, **kwargs)

        # 信号
        mag_perp_ev = (data.mean_mag_energy_density_x + data.mean_mag_energy_density_y) / J_PER_EV
        elec_z_ev = data.mean_elec_energy_density_z / J_PER_EV
        ax.plot(data.time, mag_perp_ev, ':',
                label=r'$\langle \epsilon_{B,\perp} \rangle$ (Weibel)' + label_suffix, **kwargs)
        ax.plot(data.time, elec_z_ev, ':',
                label=r'$\langle \epsilon_{E,z} \rangle$ (Two-Stream)' + label_suffix, **kwargs)

    def setup_axes(self, ax: Axes):
        ax.set_title('平均能量密度演化 (场 vs 动能)', fontsize=14)
        ax.set_ylabel(r'平均能量密度 (eV/m$^3$)', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)


class TotalEnergyPlotter(BasePlotter):
    """绘制盒子内总能量随时间的演化。"""

    def plot(self, ax: Axes, run: SimulationRun, label: str, color: Optional[str] = None, **kwargs):
        data = run.energy_data
        J_PER_EV = e

        label_suffix = f" ({label})" if label else ""

        total_kin_ev = data.total_kinetic_energy / J_PER_EV
        total_mag_ev = data.total_magnetic_energy / J_PER_EV
        total_elec_ev = data.total_electric_energy / J_PER_EV
        total_energy_ev = total_kin_ev + total_mag_ev + total_elec_ev

        ax.plot(data.time, total_kin_ev, '-', label=r'$E_{K, tot}$' + label_suffix, **kwargs)
        ax.plot(data.time, total_mag_ev, '--', label=r'$E_{B, tot}$' + label_suffix, **kwargs)
        ax.plot(data.time, total_elec_ev, '--', label=r'$E_{E, tot}$' + label_suffix, **kwargs)
        ax.plot(data.time, total_energy_ev, '-', alpha=0.7, label=r'$E_{total}$' + label_suffix, **kwargs)

    def setup_axes(self, ax: Axes):
        ax.set_title('盒子内总能量演化', fontsize=14)
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylabel('总能量 (eV)', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)