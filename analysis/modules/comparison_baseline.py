# modules/baseline_comparison.py

from typing import List, Set, Optional, Tuple

import numpy as np
from rich.prompt import Prompt
from rich.table import Table

from .base_module import BaseComparisonModule
from ..core.simulation import SimulationRun
from ..core.utils import console
from ..plotting.layout import create_analysis_figure
from ..plotting.spectrum_plotter import SpectrumComparisonPlotter
from ..plotting.styles import get_style


class BaselineSpectrumComparisonModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "基准能谱对比 (两两对比)"

    @property
    def description(self) -> str:
        return "选择一个模拟作为基准，将其他模拟的能谱与之进行两两对比，生成多张对比图。"

    @property
    def required_data(self) -> Set[str]:
        return {'initial_spectrum', 'final_spectrum'}

    def _select_baseline_run(self, runs: List[SimulationRun]) -> Optional[Tuple[SimulationRun, List[SimulationRun]]]:
        """交互式地让用户选择一个基准模拟。"""
        console.print("\n[bold]请选择一个模拟作为对比基准:[/bold]")

        table = Table(title="可作为基准的模拟")
        table.add_column("索引", justify="right", style="cyan")
        table.add_column("文件夹名称", style="magenta")
        for i, run in enumerate(runs):
            table.add_row(str(i), run.name)
        console.print(table)

        while True:
            try:
                choice_str = Prompt.ask("[bold]请输入基准模拟的索引[/bold]")
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(runs):
                    baseline_run = runs.pop(choice_idx)
                    console.print(f"[green]✔ 已选择 '{baseline_run.name}' 作为基准。[/green]")
                    return baseline_run, runs
                else:
                    console.print("[yellow]警告: 输入的索引超出范围，请重试。[/yellow]")
            except (ValueError, IndexError):
                console.print("[red]错误: 无效输入，请输入一个有效的数字索引。[/red]")

    def run(self, loaded_runs: List[SimulationRun]):
        console.print("\n[bold magenta]执行: 基准能谱对比分析...[/bold magenta]")

        valid_runs = [r for r in loaded_runs if r.initial_spectrum and r.final_spectrum]
        if len(valid_runs) < 2:
            console.print("[red]错误: 至少需要选择两个有效的模拟才能进行基准对比。[/red]")
            return

        # 1. 交互式选择基准
        selection = self._select_baseline_run(valid_runs)
        if selection is None:
            return  # 用户可能中断了操作
        baseline_run, comparison_runs = selection

        # 2. 预计算全局统一坐标轴
        console.print("  -> [cyan]正在预计算所有模拟的统一坐标轴范围...[/cyan]")
        all_runs_for_bins = [baseline_run] + comparison_runs
        try:
            bin_helper = SpectrumComparisonPlotter(all_runs_for_bins)
            common_bins = bin_helper.bins
            common_centers = bin_helper.centers
            common_widths = bin_helper.widths
        except ValueError as e:
            console.print(f"[red]错误: 创建统一能量分箱失败: {e}[/red]")
            return

        global_y_max, global_y_min = 0.0, np.inf
        for run in all_runs_for_bins:
            for spec in [run.initial_spectrum, run.final_spectrum]:
                counts, _ = np.histogram(spec.energies_MeV, bins=common_bins, weights=spec.weights)
                norm_counts = (counts / common_widths)[counts > 0]
                if norm_counts.size > 0:
                    global_y_max = max(global_y_max, norm_counts.max())
                    global_y_min = min(global_y_min, norm_counts.min())

        y_lim = (global_y_min * 0.5, global_y_max * 2.0)
        x_lim = (common_bins[0], common_bins[-1])
        console.print(f"  -> [green]统一坐标轴范围已确定。[/green]")

        # 获取当前激活的样式对象
        style = get_style()

        # 在这里定义此图表专属的、语义化的绘图参数。这些参数是基于样式基准的相对值
        lw_base = style.lw_base
        plot_params = {
            'base_initial': {'ls': style.ls_secondary, 'color': style.color_baseline_secondary, 'lw': lw_base * 3.0, 'zorder': 10},
            'base_final':   {'ls': style.ls_primary,   'color': style.color_baseline_primary,   'lw': lw_base * 2.5, 'zorder': 20},
            'comp_initial': {'ls': style.ls_tertiary,  'color': style.color_comparison_secondary, 'lw': lw_base * 2.5, 'zorder': 30},
            'comp_final':   {'ls': style.ls_primary,   'color': style.color_comparison_primary,   'lw': lw_base * 2.0, 'zorder': 40},
        }

        # 3. 循环生成每一对的对比图
        for i, comp_run in enumerate(comparison_runs):
            console.print(f"\n  ({i + 1}/{len(comparison_runs)}) 正在生成对比图: [bold]{baseline_run.name}[/bold] vs [bold]{comp_run.name}[/bold]...")

            exact_filename = f"comparison_baseline_{baseline_run.name}_vs_{comp_run.name}"
            runs_for_this_plot = [baseline_run, comp_run]

            # 2. 将精确的文件名通过 override_filename 传入
            with create_analysis_figure(
                    runs_for_this_plot,
                    "comparison_baseline",  # base_filename 仍然用于生成默认标题
                    num_plots=1,
                    figsize=(12, 8),
                    override_filename=exact_filename
            ) as (fig, ax_plot):
                # 手动覆盖布局管理器生成的标题，以获得更具体的标题
                fig.suptitle(f"能谱对比: {baseline_run.name} (基准) vs {comp_run.name}", fontsize=18)

                # --- 绘制能谱 ---
                # 基准模拟 (黑/灰色系)
                counts_base_i, _ = np.histogram(baseline_run.initial_spectrum.energies_MeV, bins=common_bins, weights=baseline_run.initial_spectrum.weights)
                dNdE_base_i = counts_base_i / common_widths
                mask_base_i = dNdE_base_i > 0
                ax_plot.plot(common_centers[mask_base_i], dNdE_base_i[mask_base_i], label=f'{baseline_run.name} (初始)', **plot_params['base_initial'])

                counts_base_f, _ = np.histogram(baseline_run.final_spectrum.energies_MeV, bins=common_bins, weights=baseline_run.final_spectrum.weights)
                dNdE_base_f = counts_base_f / common_widths
                mask_base_f = dNdE_base_f > 0
                ax_plot.plot(common_centers[mask_base_f], dNdE_base_f[mask_base_f], label=f'{baseline_run.name} (最终)', **plot_params['base_final'])

                # 对比模拟 (彩色)
                counts_comp_i, _ = np.histogram(comp_run.initial_spectrum.energies_MeV, bins=common_bins, weights=comp_run.initial_spectrum.weights)
                dNdE_comp_i = counts_comp_i / common_widths
                mask_comp_i = dNdE_comp_i > 0
                ax_plot.plot(common_centers[mask_comp_i], dNdE_comp_i[mask_comp_i], label=f'{comp_run.name} (初始)', **plot_params['comp_initial'])

                counts_comp_f, _ = np.histogram(comp_run.final_spectrum.energies_MeV, bins=common_bins, weights=comp_run.final_spectrum.weights)
                dNdE_comp_f = counts_comp_f / common_widths
                mask_comp_f = dNdE_comp_f > 0
                ax_plot.plot(common_centers[mask_comp_f], dNdE_comp_f[mask_comp_f], label=f'{comp_run.name} (最终)', **plot_params['comp_final'])

                # --- 设置坐标轴 ---
                ax_plot.set_title("初始与最终能谱对比", fontsize=16)
                ax_plot.set_xlim(x_lim)
                ax_plot.set_ylim(y_lim)
                ax_plot.set_xscale('log')
                ax_plot.set_yscale('log')
                ax_plot.set_xlabel('动能 (MeV)')
                ax_plot.set_ylabel('dN/dE [/MeV]')
                ax_plot.grid(True, which="both", ls="--", alpha=0.5)
                ax_plot.legend()
