# modules/energy_evolution.py

import matplotlib.pyplot as plt
from typing import List, Set

from .base_module import BaseAnalysisModule
from core.simulation import SimulationRun
from core.utils import console, plot_parameter_table
from scipy.constants import e


class EnergyEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "能量演化分析"

    @property
    def description(self) -> str:
        return "绘制场能（电/磁）、动能和总能量随时间的演化图。"

    @property
    def required_data(self) -> Set[str]:
        # 需要能量演化数据，以及用于参数表的初始粒子数
        return {'energy', 'initial_spectrum'}

    def run(self, loaded_runs: List[SimulationRun]):
        """为每个模拟生成能量演化图。"""
        console.print("\n[bold magenta]执行: 能量演化分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.energy_data]
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的能量数据，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            output_name = f"analysis_energy_evolution_{run.name}.png"
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_plot(run, output_name)

    def _generate_plot(self, run: SimulationRun, output_name: str):
        data = run.energy_data
        J_PER_EV = e

        kin_density_ev = data.mean_kin_energy_density / J_PER_EV
        mag_density_total_ev = data.mean_mag_energy_density_total / J_PER_EV
        elec_density_total_ev = data.mean_elec_energy_density_total / J_PER_EV

        total_kin_ev = data.total_kinetic_energy / J_PER_EV
        total_mag_ev = data.total_magnetic_energy / J_PER_EV
        total_elec_ev = data.total_electric_energy / J_PER_EV

        fig, (ax_density, ax_total, ax_table) = plt.subplots(
            3, 1, figsize=(12, 18),
            gridspec_kw={'height_ratios': [4, 4, 3]}, constrained_layout=True
        )
        fig.suptitle(f"能量演化分析: {run.name}", fontsize=18, y=1.02)

        # 子图1: 平均能量密度
        ax_density.set_title('平均能量密度演化 (场 vs 动能)', fontsize=14)
        ax_density.plot(data.time, kin_density_ev, '-', color='black', lw=2.5, label=r'$\langle \epsilon_K \rangle$ (动能)')
        ax_density.plot(data.time, mag_density_total_ev, '--', color='purple', lw=2, label=r'$\langle \epsilon_B \rangle$ (总磁能)')
        ax_density.plot(data.time, elec_density_total_ev, '--', color='orange', lw=2, label=r'$\langle \epsilon_E \rangle$ (总电能)')

        # 信号分析
        mag_perp_ev = data.mean_mag_energy_density_x / J_PER_EV + data.mean_mag_energy_density_y / J_PER_EV
        elec_z_ev = data.mean_elec_energy_density_z / J_PER_EV
        ax_density.plot(data.time, mag_perp_ev, ':', color='red', lw=2, label=r'$\langle \epsilon_{B,\perp} \rangle$ (Weibel Signal)')
        ax_density.plot(data.time, elec_z_ev, ':', color='cyan', lw=2, label=r'$\langle \epsilon_{E,z} \rangle$ (Two-Stream Signal)')

        ax_density.set_ylabel(r'平均能量密度 (eV/m$^3$)', fontsize=12)
        ax_density.set_yscale('log')
        ax_density.legend(fontsize=11)

        # 子图2: 总能量
        ax_total.set_title('盒子内总能量演化', fontsize=14)
        ax_total.plot(data.time, total_kin_ev, '-', color='black', lw=2.5, label=r'$E_{K, tot}$')
        ax_total.plot(data.time, total_mag_ev, '--', color='purple', lw=2.5, label=r'$E_{B, tot}$')
        ax_total.plot(data.time, total_elec_ev, '--', color='orange', lw=2.5, label=r'$E_{E, tot}$')
        total_energy_ev = total_kin_ev + total_mag_ev + total_elec_ev
        ax_total.plot(data.time, total_energy_ev, '-', color='gray', lw=2, alpha=0.8, label=r'$E_{total}$')
        ax_total.set_xlabel('时间 (s)', fontsize=12)
        ax_total.set_ylabel('总能量 (eV)', fontsize=12)
        ax_total.set_yscale('log')
        ax_total.legend(fontsize=11)

        # 子图3: 参数表
        plot_parameter_table(ax_table, run)

        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        console.print(f"  [green]✔ 图已保存: {output_name}[/green]")