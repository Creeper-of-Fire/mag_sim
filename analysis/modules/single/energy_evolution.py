# modules/energy_evolution.py

from typing import List

from scipy.constants import e

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import AnalysisLayout


class EnergyEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能量演化分析"

    @property
    def description(self) -> str:
        return "绘制场能（电/磁）、动能和总能量随时间的演化图。"

    def run(self, loaded_runs: List[SimulationRun]):
        """为每个模拟生成能量演化图。"""
        console.print("\n[bold magenta]执行: 能量演化分析...[/bold magenta]")

        valid_runs = filter_valid_runs(
            loaded_runs,
            require_particles=True,
            min_particle_files=2,
            require_fields=True,
            min_field_files=2
        )
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的能量数据，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run)

    @staticmethod
    def _generate_single_run_plot(run: SimulationRun):
        with AnalysisLayout(run, "analysis_energy_evolution") as layout:
            ax_density = layout.request_axes()
            label = run.name
            data = run.energy_data
            J_PER_EV = e
            label_suffix = f" ({label})" if label else ""

            # 动能
            ax_density.plot(data.time, data.mean_kin_energy_density / J_PER_EV,
                            label=r'$\langle \epsilon_K \rangle$' + label_suffix)

            # 磁能
            ax_density.plot(data.time, data.mean_mag_energy_density_total / J_PER_EV, '--',
                            label=r'$\langle \epsilon_B \rangle$' + label_suffix)

            # 电能
            ax_density.plot(data.time, data.mean_elec_energy_density_total / J_PER_EV, '--',
                            label=r'$\langle \epsilon_E \rangle$' + label_suffix)

            # 信号
            mag_perp_ev = (data.mean_mag_energy_density_x + data.mean_mag_energy_density_y) / J_PER_EV
            elec_z_ev = data.mean_elec_energy_density_z / J_PER_EV
            ax_density.plot(data.time, mag_perp_ev, ':',
                            label=r'$\langle \epsilon_{B,\perp} \rangle$ (Weibel)' + label_suffix)
            ax_density.plot(data.time, elec_z_ev, ':',
                            label=r'$\langle \epsilon_{E,z} \rangle$ (Two-Stream)' + label_suffix)

            ax_density.set_ylabel(r'平均能量密度 (eV/m$^3$)', fontsize=12)
            ax_density.set_yscale('log')
            ax_density.legend(fontsize=11)
            ax_density.grid(True, which="both", ls="--", alpha=0.5)

            ax_total = layout.request_axes()

            total_kin_ev = data.total_kinetic_energy / J_PER_EV
            total_mag_ev = data.total_magnetic_energy / J_PER_EV
            total_elec_ev = data.total_electric_energy / J_PER_EV
            total_energy_ev = total_kin_ev + total_mag_ev + total_elec_ev

            ax_total.plot(data.time, total_kin_ev, '-', label=r'$E_{K, tot}$' + label_suffix)
            ax_total.plot(data.time, total_mag_ev, '--', label=r'$E_{B, tot}$' + label_suffix)
            ax_total.plot(data.time, total_elec_ev, '--', label=r'$E_{E, tot}$' + label_suffix)
            ax_total.plot(data.time, total_energy_ev, '-', alpha=0.7, label=r'$E_{total}$' + label_suffix)

            ax_total.set_xlabel('时间 (s)', fontsize=12)
            ax_total.set_ylabel('总能量 (eV)', fontsize=12)
            ax_total.set_yscale('log')
            ax_total.legend(fontsize=11)
            ax_total.grid(True, which="both", ls="--", alpha=0.5)
