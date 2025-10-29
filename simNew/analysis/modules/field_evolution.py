# modules/field_evolution.py

import matplotlib.pyplot as plt
from typing import List, Set

from .base_module import BaseAnalysisModule
from core.simulation import SimulationRun
from core.utils import console, plot_parameter_table


class FieldEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "磁场演化分析"

    @property
    def description(self) -> str:
        return "绘制磁场分量的RMS/平均值以及总场强随时间的演化。"

    @property
    def required_data(self) -> Set[str]:
        # 只需要场演化数据，以及参数表数据
        return {'field', 'initial_spectrum'}

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 磁场演化分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.field_data]
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的场数据，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            output_name = f"analysis_field_evolution_{run.name}.png"
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_plot(run, output_name)

    def _generate_plot(self, run: SimulationRun, output_name: str):
        fig, (ax_anisotropy, ax_comp, ax_mag, ax_table) = plt.subplots(
            4, 1, figsize=(12, 18),
            gridspec_kw={'height_ratios': [3, 3, 3, 3]}, constrained_layout=True
        )

        # 子图1: RMS (湍流各向异性)
        ax_anisotropy.set_title('磁场分量RMS值演化 (湍流各向异性分析)', fontsize=14)
        ax_anisotropy.plot(run.field_data.time, run.field_data.b_rms_x_normalized, '-', color='red', lw=2, label='RMS(Bx)')
        ax_anisotropy.plot(run.field_data.time, run.field_data.b_rms_y_normalized, '--', color='green', lw=2, label='RMS(By)')
        ax_anisotropy.plot(run.field_data.time, run.field_data.b_rms_z_normalized, ':', color='blue', lw=2, label='RMS(Bz)')
        ax_anisotropy.set_ylabel('分量RMS值 / B_norm')
        ax_anisotropy.set_yscale('log')
        ax_anisotropy.legend()

        # 子图2: 平均值 (宏观各向异性)
        ax_comp.set_title('磁场分量平均值 <B> 演化 (宏观各向异性分析)', fontsize=14)
        ax_comp.plot(run.field_data.time, run.field_data.b_mean_x_normalized, '-', color='red', lw=2, label='<Bx>')
        ax_comp.plot(run.field_data.time, run.field_data.b_mean_y_normalized, '--', color='green', lw=2, label='<By>')
        ax_comp.plot(run.field_data.time, run.field_data.b_mean_z_normalized, ':', color='blue', lw=2, label='<Bz>')
        ax_comp.axhline(0.0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_comp.set_ylabel('平均磁场分量 <B_i> / B_norm')
        ax_comp.legend()

        # 子图3: 强度 (增长与饱和)
        ax_mag.set_title('磁场强度演化 (增长与饱和分析)', fontsize=14)
        ax_mag.plot(run.field_data.time, run.field_data.b_mean_abs_normalized, '-', color='purple', lw=2.5, label='<|B|> / B_norm')
        ax_mag.plot(run.field_data.time, run.field_data.b_max_normalized, '--', color='orange', lw=2, alpha=0.9, label='Max|B| / B_norm')
        ax_mag.set_xlabel('时间 (s)', fontsize=12)
        ax_mag.set_ylabel('磁场强度 |B| / B_norm')
        ax_mag.set_yscale('log')
        ax_mag.legend()

        # 子图4: 参数表
        plot_parameter_table(ax_table, run)

        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        console.print(f"  [green]✔ 图已保存: {output_name}[/green]")