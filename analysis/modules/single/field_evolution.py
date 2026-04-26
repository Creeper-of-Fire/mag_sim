# modules/field_evolution.py

from typing import List

from analysis.core.simulation import SimulationRun
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils.spectrum_tools import filter_valid_runs
from analysis.plotting.layout import AnalysisLayout


class FieldEvolutionModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "磁场演化分析"

    @property
    def description(self) -> str:
        return "绘制磁场分量的RMS/平均值以及总场强随时间的演化。"

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 磁场演化分析...[/bold magenta]")

        valid_runs = filter_valid_runs(
            loaded_runs,
            require_fields=True,
            min_field_files=2
        )
        if not valid_runs:
            console.print("[yellow]警告: 没有加载到有效的场数据，跳过此分析。[/yellow]")
            return

        for i, run in enumerate(valid_runs):
            console.print(f"  ({i + 1}/{len(valid_runs)}) 正在绘制 [bold]{run.name}[/bold]...")
            self._generate_single_run_plot(run)

    def _generate_single_run_plot(self, run: SimulationRun):
        with AnalysisLayout(run, "analysis_field_evolution") as layout:
            data = run.field_data
            label_suffix = f" ({run.name})" if run.name else ""
            ax_rms = layout.request_axes()
            ax_mean = layout.request_axes()
            ax_mag = layout.request_axes()

            ax_rms.plot(data.time, data.b_rms_x_normalized, '-', label='RMS(Bx)' + label_suffix)
            ax_rms.plot(data.time, data.b_rms_y_normalized, '--', label='RMS(By)' + label_suffix)
            ax_rms.plot(data.time, data.b_rms_z_normalized, ':', label='RMS(Bz)' + label_suffix)

            ax_rms.set_title('磁场分量RMS值演化 (湍流各向异性分析)', fontsize=14)
            ax_rms.set_ylabel('分量RMS值 / B_norm')
            ax_rms.set_yscale('log')
            ax_rms.legend(loc='best', fontsize='small')
            ax_rms.grid(True, which="both", ls="--", alpha=0.5)

            ax_mean.plot(data.time, data.b_mean_x_normalized, '-', label='<Bx>' + label_suffix)
            ax_mean.plot(data.time, data.b_mean_y_normalized, '--', label='<By>' + label_suffix)
            ax_mean.plot(data.time, data.b_mean_z_normalized, ':', label='<Bz>' + label_suffix)
            ax_mean.axhline(0.0, color='black', linestyle='-', linewidth=1, alpha=0.7)

            ax_mag.set_title('磁场分量平均值 <B> 演化 (宏观各向异性分析)', fontsize=14)
            ax_mag.set_ylabel('平均磁场分量 <B_i> / B_norm')
            ax_mag.legend(loc='best', fontsize='small')
            ax_mag.grid(True, which="both", ls="--", alpha=0.5)

            ax_mag.plot(data.time, data.b_mean_abs_normalized, '-', label='<|B|>' + label_suffix)
            ax_mag.plot(data.time, data.b_max_normalized, '--', alpha=0.9, label='Max|B|' + label_suffix)

            ax_mag.set_title('磁场强度演化 (增长与饱和分析)', fontsize=14)
            ax_mag.set_xlabel('时间 (s)', fontsize=12)
            ax_mag.set_ylabel('磁场强度 |B| / B_norm')
            ax_mag.set_yscale('log')
            ax_mag.legend(loc='best', fontsize='small')
            ax_mag.grid(True, which="both", ls="--", alpha=0.5)
