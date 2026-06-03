# analysis/modules/single/tail_statistics_timeseries.py
"""
时序演化版 tail_statistics — 单 run 分析。

对每个 run 独立出图，X=物理时间，子图与 tail_statisticsV2 一致。
"""

import asyncio
from typing import List, Optional

import numpy as np

from analysis.core.data_loader import _get_step_from_filename
from analysis.core.group_merger import detect_and_merge_groups
from analysis.core.simulation import SimulationRun
from analysis.core.simulationGroup import SimulationRunGroup
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseAnalysisModule
from analysis.modules.utils.time_series import (
    GroupedStepMetrics, AggregatedMetrics,
    extract_tail_time_series_async, extract_grouped_time_series_async,
    compute_run_avg_last_n, avg_last_n,
)
from analysis.plotting.data_layout import DataLayout
from analysis.plotting.styles import get_style


class TailStatisticsTimeSeriesSingleModule(BaseAnalysisModule):
    @property
    def name(self) -> str:
        return "时序演化：高能尾部分析 (Time-Averaged Tail)"

    @property
    def description(self) -> str:
        return "遍历每个模拟的完整时间演化，对末尾N步做平均。输出时序图(X=时间)。"

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

    async def _extract(self, run: SimulationRun):
        singles = self._unpack_runs(run)
        if not singles or len(singles[0].particle_files) <= 2:
            return None, None
        console.print(f"    [{run.name}] {len(singles[0].particle_files)} 时间步")
        if len(singles) == 1:
            series = await extract_tail_time_series_async(singles[0], self.INTERVALS, field_files_needed=True)
            aggregated = await compute_run_avg_last_n(singles[0], self.INTERVALS, n=self.n_avg, field_files_needed=True)
            return series, aggregated
        series = await extract_grouped_time_series_async(singles, self.INTERVALS, field_files_needed=True)
        aggregated = avg_last_n(series, n=self.n_avg) if series else None
        return series, aggregated

    def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print(f"\n[bold magenta]执行: {self.name}...[/bold magenta]")

        # 分组合并步骤（类似 compare 模块使用 ComparisonContext 的模式）
        merged_runs = detect_and_merge_groups(loaded_runs)

        async def _process_all():
            tasks = [self._extract(run) for run in merged_runs]
            return list(await asyncio.gather(*tasks))

        raw_results = asyncio.get_event_loop().run_until_complete(_process_all())

        for run, (series, aggregated) in zip(merged_runs, raw_results):
            if series is None or aggregated is None:
                console.print(f"  [yellow]跳过 {run.name}（时间步不足）[/yellow]")
                continue
            self._plot_timeseries(run, series, aggregated, style)

        console.print("[bold green]时序分析完成。[/bold green]")

    # =========================================================================
    # 时序演化图（每个 run 独立，X = 时间）
    # =========================================================================

    def _plot_timeseries(self, run: SimulationRun, series, avg: AggregatedMetrics, style):
        singles = self._unpack_runs(run)
        is_grouped = isinstance(series[0], GroupedStepMetrics)

        dt = getattr(singles[0].sim, 'dt', 0.0)
        step_numbers = [_get_step_from_filename(f) or i
                        for i, f in enumerate(singles[0].particle_files[:len(series)])]
        if dt > 0:
            x_time = np.array(step_numbers) * dt
            time_label = "时间 (s)"
        else:
            x_time = np.array(step_numbers, dtype=float)
            time_label = "时间步"

        # 图1：各能段 excess_ratio
        with DataLayout(run, f"tail_ts_excess_{run.name}", plot_ratio=(10, 3.5), ncols=2,
                        shared_xlabel=time_label) as layout:
            for low, high in self.INTERVALS:
                ax = layout.request_axes()
                label = f"{low:.2f}-{high:.2f}" if high else f"{low:.2f}-inf"
                vals, th_unc, run_std = self._extract_fields(series, label, is_grouped)

                ax.fill_between(x_time, (vals - th_unc) * 100, (vals + th_unc) * 100,
                                color=style.color_comparison_primary, alpha=0.08, label='理论传递误差')
                if run_std is not None:
                    ax.fill_between(x_time, (vals - run_std) * 100, (vals + run_std) * 100,
                                    color=style.color_comparison_secondary, alpha=0.10, label='跨 run 统计散布')

                ax.plot(x_time, vals * 100, color=style.color_comparison_primary, lw=style.lw_base)

                if label in avg.tail_excess:
                    avg_val = avg.tail_excess[label] * 100
                    ax.axhline(avg_val, color=style.color_comparison_primary, linestyle='--', alpha=0.6, lw=1,
                               label=f'末尾{self.n_avg}步平均 = {avg_val:.2f}%')
                    if len(series) > self.n_avg:
                        ax.axvspan(x_time[-self.n_avg], x_time[-1], alpha=0.04,
                                   color=style.color_comparison_primary)

                band_str = f"${low:.2f}T < E < {high:.2f}T$" if high else f"$E > {low:.2f}T$"
                ax.text(0.98, 0.95, band_str, transform=ax.transAxes, ha='right', va='top',
                        fontsize='medium', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                ax.axhline(0, color='black', linestyle='-', alpha=0.3, lw=0.8)
                ax.set_ylabel("超额能量占比 (%)")
                ax.legend(fontsize='x-small', loc='best', ncol=2)
                ax.grid(True, linestyle=':', alpha=0.4)

        # 图2：其他指标（温度、场能、能量密度）
        with DataLayout(run, f"tail_ts_other_{run.name}", plot_ratio=(10, 3.5), ncols=2,
                        shared_xlabel=time_label) as layout:
            # --- 温度 ---
            ax_t = layout.request_axes()
            temps = np.array([s.T_keV for s in series])
            ax_t.plot(x_time, temps, color=style.color_comparison_secondary, lw=style.lw_base, label='$T_{eff}$')
            if is_grouped:
                t_std = np.array([s.T_keV_std for s in series])
                ax_t.fill_between(x_time, temps - t_std, temps + t_std,
                                  color=style.color_comparison_secondary, alpha=0.10, label='跨 run 统计散布')
            ax_t.axhline(avg.T_keV, color=style.color_comparison_secondary, linestyle='--', alpha=0.6, lw=1,
                         label=f'末尾{self.n_avg}步平均 = {avg.T_keV:.2f} keV')
            ax_t.set_ylabel("$T_{eff}$ (keV)")
            ax_t.legend(fontsize='small', loc='best')
            ax_t.grid(True, linestyle='--', alpha=0.5)

            # --- 磁能占比 ---
            ax_mag = layout.request_axes()
            mag_vals = np.array([s.mag_fraction * 100 for s in series])
            ax_mag.plot(x_time, mag_vals, color='darkorange', lw=style.lw_base, label='磁能占比 $E_B$')
            if is_grouped:
                mag_std = np.array([s.mag_fraction_std * 100 for s in series])
                ax_mag.fill_between(x_time, mag_vals - mag_std, mag_vals + mag_std, color='darkorange', alpha=0.10)
            ax_mag.axhline(avg.mag_fraction * 100, color='darkorange', linestyle='--', alpha=0.6, lw=1,
                           label=f'末尾{self.n_avg}步平均')
            ax_mag.set_ylabel(r"$E_B / E_{total}$ (%)")
            ax_mag.legend(fontsize='small', loc='best')
            ax_mag.grid(True, linestyle=':', alpha=0.4)

            # --- 电能占比 ---
            ax_elec = layout.request_axes()
            elec_vals = np.array([s.elec_fraction * 100 for s in series])
            ax_elec.plot(x_time, elec_vals, color='crimson', lw=style.lw_base, label='电能占比 $E_E$')
            ax_elec.set_ylabel(r"$E_E / E_{total}$ (%)")
            ax_elec.legend(fontsize='small', loc='best')
            ax_elec.grid(True, alpha=0.3)

            # --- 总场能 ---
            ax_s = layout.request_axes()
            field_vals = np.array([s.field_fraction * 100 for s in series])
            ax_s.plot(x_time, field_vals, color='indigo', lw=style.lw_base, label='总场能 (B+E)')
            ax_s.set_ylabel(r"$(E_B + E_E) / E_{total}$ (%)")
            ax_s.legend(fontsize='small', loc='best')
            ax_s.grid(True, alpha=0.3)

            # --- 能量密度 ---
            for attr, color, ylabel in [
                ('kin_density_norm', 'steelblue', r"$U_{kin} / (n_0 m_e c^2)$"),
                ('mag_density_norm', 'darkorange', r"$U_B / (n_0 m_e c^2)$"),
                ('elec_density_norm', 'crimson', r"$U_E / (n_0 m_e c^2)$"),
                ('total_density_norm', 'darkgreen', r"$U_{total} / (n_0 m_e c^2)$"),
            ]:
                ax_d = layout.request_axes()
                d_vals = np.array([getattr(s, attr) for s in series])
                ax_d.plot(x_time, d_vals, color=color, lw=style.lw_base)
                if is_grouped:
                    d_std = np.array([getattr(s, attr + '_std') for s in series])
                    ax_d.fill_between(x_time, d_vals - d_std, d_vals + d_std, color=color, alpha=0.10)
                ax_d.set_ylabel(ylabel)
                ax_d.grid(True, linestyle=':', alpha=0.4)
                ax_d.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    @staticmethod
    def _extract_fields(series, label: str, is_grouped: bool):
        vals = np.array([s.tail_excess.get(label, 0.0) for s in series])
        th_unc = np.array([s.tail_uncertainty.get(label, 0.0) for s in series])
        run_std = None
        if is_grouped:
            run_std = np.array([s.tail_excess_std.get(label, 0.0) for s in series])
        return vals, th_unc, run_std
