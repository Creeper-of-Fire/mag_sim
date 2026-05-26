# analysis/modules/compare/tail_statistics_timeseries.py
"""
时序演化版 tail_statisticsV2 — 参数扫描对比。

跨 run 对比，X=参数（ParameterSelector 交互选择），Y=时间平均指标。
分组、X 轴选择均由 ComparisonContext 处理，与现有对比模块一致。

时序演化（图A，每个 run 独立出图）已拆分至 single/tail_statistics_timeseries.py。
"""

import asyncio
from typing import List, Optional

import numpy as np

from analysis.core.simulation import SimulationRun
from analysis.core.simulationGroup import SimulationRunGroup
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils.time_series import (
    AggregatedMetrics,
    extract_tail_time_series_async, extract_grouped_time_series_async,
    compute_run_avg_last_n, avg_last_n,
)
from analysis.plotting.comparison_layout import ComparisonContext, ComparisonLayout
from analysis.plotting.styles import get_style


class TailStatisticsTimeSeriesModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "时序演化：参数扫描对比 (Time-Averaged Tail Scan)"

    @property
    def description(self) -> str:
        return "跨 run 对比时间平均尾部指标，X=参数。时序演化图请使用对应的 single 模块。"

    INTERVALS = [
        (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0),
        (3.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, None),
    ]

    def __init__(self, n_avg: int = 5):
        super().__init__()
        self.n_avg = n_avg

    @staticmethod
    def _unpack_runs(run: SimulationRun):
        if isinstance(run, SimulationRunSingle):
            return [run]
        if isinstance(run, SimulationRunGroup):
            return run.runs
        return []

    async def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print(f"\n[bold magenta]执行: {self.name}...[/bold magenta]")

        ctx = ComparisonContext(loaded_runs, "tail_timeseries")
        runs, x_scaled = ctx.unpack
        x_raw = ctx.x_raw
        x_label_key = ctx.x_label_key

        async def _extract(index: int, run: SimulationRun):
            singles = self._unpack_runs(run)
            if not singles or len(singles[0].particle_files) <= 2:
                return None
            console.print(f"    [{run.name}] {x_label_key}={x_raw[index]} — {len(singles[0].particle_files)} 时间步")
            if len(singles) == 1:
                series = await extract_tail_time_series_async(singles[0], self.INTERVALS, field_files_needed=True)
                return await compute_run_avg_last_n(singles[0], self.INTERVALS, n=self.n_avg, field_files_needed=True)
            series = await extract_grouped_time_series_async(singles, self.INTERVALS, field_files_needed=True)
            return avg_last_n(series, n=self.n_avg) if series else None

        console.print(f"  并行提取 {len(runs)} 个 run 的时序数据...")
        raw_results = list(await asyncio.gather(*[_extract(i, run) for i, run in enumerate(runs)]))

        all_aggregated: List[Optional[AggregatedMetrics]] = []
        for i, agg in enumerate(raw_results):
            all_aggregated.append(agg)
            if agg is None:
                console.print(f"  [yellow]跳过 {runs[i].name}（时间步不足）[/yellow]")

        valid = [(i, a) for i, a in enumerate(all_aggregated) if a is not None]
        if not valid:
            console.print("[yellow]没有有效的时序数据。[/yellow]")
            return

        self._plot_param_scan(ctx, runs, x_scaled, x_raw, x_label_key,
                              [i for i, _ in valid], [a for _, a in valid], style)

        console.print("[bold green]参数扫描分析完成。[/bold green]")

    # =========================================================================
    # 参数扫描对比（X = 参数，Y = 时间平均值）
    # =========================================================================

    def _plot_param_scan(self, ctx: ComparisonContext, runs, x_scaled, x_raw, x_label_key,
                         valid_indices: List[int], all_aggregated: List[AggregatedMetrics], style):
        x_valid = x_scaled[valid_indices]

        # 图1：各能段 excess_ratio 参数扫描
        with ComparisonLayout(ctx, suffix="excess", plot_ratio=(10, 3.5), ncols=2) as layout:
            for low, high in self.INTERVALS:
                ax = layout.request_axes()
                label = f"{low:.2f}-{high:.2f}" if high else f"{low:.2f}-inf"

                avg_vals = np.array([a.tail_excess.get(label, 0.0) * 100 for a in all_aggregated])
                th_unc = np.array([a.tail_uncertainty.get(label, 0.0) * 100 for a in all_aggregated])
                run_std = np.array([a.tail_excess_std.get(label, 0.0) * 100 for a in all_aggregated])

                combined = np.sqrt(th_unc ** 2 + run_std ** 2)

                ax.fill_between(x_valid, avg_vals - combined, avg_vals + combined,
                                color=style.color_comparison_primary, alpha=0.1,
                                label=f'合成误差 (末尾{self.n_avg}步)')

                ax.errorbar(x_valid, avg_vals, yerr=run_std if np.any(run_std > 0) else None,
                            fmt='-o', capsize=4, elinewidth=1.5,
                            color=style.color_comparison_primary,
                            label=f'末尾{self.n_avg}步时间平均')

                band_str = f"${low:.2f}T < E < {high:.2f}T$" if high else f"$E > {low:.2f}T$"
                ax.text(0.98, 0.95, band_str, transform=ax.transAxes, ha='right', va='top',
                        fontsize='medium', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                ax.axhline(0, color='black', linestyle='-', alpha=0.3, lw=1)
                ax.set_ylabel("超额能量占比 (%)")
                ax.legend(fontsize='x-small', loc='upper left', ncol=2)
                ax.grid(True, linestyle=':', alpha=0.4)

        # 图2：其他指标参数扫描（温度、场能、能量密度）
        with ComparisonLayout(ctx, suffix="other", plot_ratio=(10, 3.5), ncols=2) as layout:
            # --- 温度 ---
            ax_temp = layout.request_axes()
            avg_temps = [a.T_keV for a in all_aggregated]
            avg_temp_err = [a.T_keV_std for a in all_aggregated]
            ax_temp.errorbar(x_valid, avg_temps,
                             yerr=np.array(avg_temp_err) if np.any(np.array(avg_temp_err) > 0) else None,
                             fmt='-s', capsize=4, color=style.color_comparison_secondary, lw=style.lw_base,
                             label=f'末尾{self.n_avg}步平均温度')
            ax_temp.set_ylabel("$T_{eff}$ (keV)")
            ax_temp.legend(fontsize='small', loc='best')
            ax_temp.grid(True, linestyle='--', alpha=0.5)

            # --- 磁能占比 ---
            ax_mag = layout.request_axes()
            avg_mag = [a.mag_fraction for a in all_aggregated]
            avg_mag_err = [a.mag_fraction_std for a in all_aggregated]
            ax_mag.errorbar(x_valid, avg_mag,
                            yerr=np.array(avg_mag_err) if np.any(np.array(avg_mag_err) > 0) else None,
                            fmt='-d', capsize=4, color='darkorange', label='磁能占比 $E_B$')
            ax_mag.set_ylabel(r"$E_B / E_{total}$")
            ax_mag.legend(fontsize='small', loc='best')

            # --- 电能占比 ---
            ax_elec = layout.request_axes()
            ax_elec.errorbar(x_valid, [a.elec_fraction for a in all_aggregated],
                             yerr=np.array([a.elec_fraction_std for a in all_aggregated]) if np.any(
                                 np.array([a.elec_fraction_std for a in all_aggregated]) > 0) else None,
                             fmt='-^', capsize=4, color='crimson', label='电能占比 $E_E$')
            ax_elec.set_ylabel(r"$E_E / E_{total}$")
            ax_elec.legend(fontsize='small')
            ax_elec.grid(True, alpha=0.3)

            # --- 总场能 ---
            ax_s = layout.request_axes()
            ax_s.errorbar(x_valid, [a.mag_fraction + a.elec_fraction for a in all_aggregated],
                          fmt='-P', color='indigo', label='总场能 (B+E)')
            ax_s.set_ylabel(r"$(E_B + E_E) / E_{total}$")

            # --- 能量密度 ---
            for attr, color, ylabel in [
                ('kin_density_norm', 'steelblue', r"$U_{kin} / (n_0 m_e c^2)$"),
                ('mag_density_norm', 'darkorange', r"$U_B / (n_0 m_e c^2)$"),
                ('elec_density_norm', 'crimson', r"$U_E / (n_0 m_e c^2)$"),
                ('total_density_norm', 'darkgreen', r"$U_{total} / (n_0 m_e c^2)$"),
            ]:
                ax_e = layout.request_axes()
                avg_vals = [getattr(a, attr) for a in all_aggregated]
                avg_errs = [getattr(a, attr + '_std') for a in all_aggregated]
                ax_e.errorbar(x_valid, avg_vals,
                              yerr=np.array(avg_errs) if np.any(np.array(avg_errs) > 0) else None,
                              fmt='-o' if 'kin' in attr else ('-d' if 'mag' in attr else ('-^' if 'elec' in attr else '-s')),
                              capsize=4, color=color, linewidth=2 if 'total' in attr else style.lw_base,
                              label=ylabel.split('/')[0].strip('$\\'))
                ax_e.set_ylabel(ylabel)
                ax_e.legend(fontsize='small', loc='best')
                ax_e.grid(True, linestyle=':', alpha=0.4)
                ax_e.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
