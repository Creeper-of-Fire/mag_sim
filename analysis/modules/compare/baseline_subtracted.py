# analysis/modules/compare/baseline_subtracted.py
"""
基准扣除分析 — 用参数=0 的模拟作为基准，计算其他模拟的 excess 差值。

消除系统性数值偏移（如温度自拟合导致的固定偏向），分离磁场贡献的纯信号。

输出：
  图1: 参数扫描 — 差分 excess vs σ（基准点强制过零）
  图2: 每个非基准 run 的时序差分图 — Δexcess vs 时间
"""

import asyncio
from typing import List, Optional, Dict

import numpy as np

from analysis.core.data_loader import _get_step_from_filename
from analysis.core.simulation import SimulationRun
from analysis.core.simulationGroup import SimulationRunGroup
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.core.utils import console
from analysis.modules.abstract.base_module import BaseComparisonModule
from analysis.modules.utils.time_series import (
    GroupedStepMetrics, AggregatedMetrics,
    extract_tail_time_series_async, extract_grouped_time_series_async,
    compute_run_avg_last_n, avg_last_n,
)
from analysis.plotting.comparison_layout import ComparisonContext, ComparisonLayout
from analysis.plotting.data_layout import DataLayout
from analysis.plotting.styles import get_style


class BaselineSubtractedModule(BaseComparisonModule):
    @property
    def name(self) -> str:
        return "基准扣除：参数=0 差分 (Baseline Subtracted)"

    @property
    def description(self) -> str:
        return "用参数=0 的模拟作为基准，计算其他模拟的 excess_ratio 差值，消除系统性数值偏移。"

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

    async def _extract_series_and_agg(self, run: SimulationRun):
        singles = self._unpack_runs(run)
        if not singles or len(singles[0].particle_files) <= 2:
            return None, None
        if len(singles) == 1:
            series = await extract_tail_time_series_async(
                singles[0], self.INTERVALS, field_files_needed=True)
            agg = await compute_run_avg_last_n(
                singles[0], self.INTERVALS, n=self.n_avg, field_files_needed=True)
        else:
            series = await extract_grouped_time_series_async(
                singles, self.INTERVALS, field_files_needed=True)
            agg = avg_last_n(series, n=self.n_avg) if series else None
        return series, agg

    @staticmethod
    def _find_baseline_index(x_raw: list[str]) -> Optional[int]:
        for i, x in enumerate(x_raw):
            try:
                if float(x) == 0.0:
                    return i
            except ValueError:
                continue
        return None

    @staticmethod
    def _subtract_agg(target: AggregatedMetrics,
                      baseline: AggregatedMetrics) -> Dict[str, float]:
        return {
            label: target.tail_excess.get(label, 0.0) - baseline.tail_excess.get(label, 0.0)
            for label in target.tail_excess
        }

    @staticmethod
    def _subtract_agg_uncertainty(target: AggregatedMetrics,
                                  baseline: AggregatedMetrics) -> Dict[str, float]:
        unc = {}
        for label in target.tail_excess:
            t_th = target.tail_uncertainty.get(label, 0.0)
            b_th = baseline.tail_uncertainty.get(label, 0.0)
            t_std = target.tail_excess_std.get(label, 0.0)
            b_std = baseline.tail_excess_std.get(label, 0.0)
            unc[label] = np.sqrt(t_th**2 + b_th**2 + t_std**2 + b_std**2)
        return unc

    async def run(self, loaded_runs: List[SimulationRun]):
        style = get_style()
        console.print(f"\n[bold magenta]执行: {self.name}...[/bold magenta]")

        ctx = ComparisonContext(loaded_runs, "baseline_subtracted", min_runs=2)
        runs, x_scaled = ctx.unpack
        x_raw = ctx.x_raw
        x_label_key = ctx.x_label_key

        baseline_idx = self._find_baseline_index(x_raw)
        if baseline_idx is None:
            console.print("[red]找不到参数=0 的基准 run。请确保包含无磁场模拟。[/red]")
            return
        console.print(f"  基准: {runs[baseline_idx].name} ({x_label_key}={x_raw[baseline_idx]})")

        console.print(f"  并行提取 {len(runs)} 个 run 的时序数据...")
        results = list(await asyncio.gather(*[
            self._extract_series_and_agg(run) for run in runs
        ]))

        baseline_series, baseline_agg = results[baseline_idx]
        if baseline_agg is None:
            console.print("[red]基准 run 数据不足。[/red]")
            return

        # ---- 图1: 参数扫描 ----
        valid_indices: List[int] = []
        diff_excess: List[Dict[str, float]] = []
        diff_unc: List[Dict[str, float]] = []

        for i, (series, agg) in enumerate(results):
            if i == baseline_idx:
                continue
            if agg is None:
                console.print(f"  [yellow]跳过 {runs[i].name}（数据不足）[/yellow]")
                continue
            valid_indices.append(i)
            diff_excess.append(self._subtract_agg(agg, baseline_agg))
            diff_unc.append(self._subtract_agg_uncertainty(agg, baseline_agg))

        if not valid_indices:
            console.print("[yellow]没有有效的非基准 run。[/yellow]")
            return

        self._plot_param_scan(ctx, x_scaled, x_raw, x_label_key,
                              baseline_idx, valid_indices,
                              diff_excess, diff_unc, style)

        # ---- 图2: 每个非基准 run 的时序差分 ----
        for idx in valid_indices:
            series, _ = results[idx]
            if series is None or baseline_series is None:
                continue
            self._plot_timeseries_diff(
                runs[idx], series, baseline_series, style,
                x_raw[idx], x_raw[baseline_idx])

        console.print("[bold green]基准扣除分析完成。[/bold green]")

    # =========================================================================
    # 参数扫描：Δexcess vs σ
    # =========================================================================

    def _plot_param_scan(self, ctx, x_scaled, x_raw, x_label_key,
                         baseline_idx, valid_indices,
                         diff_excess, diff_unc, style):
        x_base = float(x_scaled[baseline_idx])
        x_valid = x_scaled[valid_indices]
        x_all = np.concatenate([[x_base], x_valid])

        with ComparisonLayout(ctx, suffix="diff_excess",
                              plot_ratio=(10, 3.5), ncols=2) as layout:
            for low, high in self.INTERVALS:
                ax = layout.request_axes()
                label = f"{low:.2f}-{high:.2f}" if high else f"{low:.2f}-inf"

                vals = np.array([d.get(label, 0.0) * 100 for d in diff_excess])
                uncs = np.array([u.get(label, 0.0) * 100 for u in diff_unc])

                vals_all = np.concatenate([[0.0], vals])
                uncs_all = np.concatenate([[0.0], uncs])

                ax.fill_between(x_all, vals_all - uncs_all, vals_all + uncs_all,
                                color=style.color_comparison_primary, alpha=0.1,
                                label='合成误差')
                ax.errorbar(x_all, vals_all,
                            yerr=uncs_all if np.any(uncs_all > 0) else None,
                            fmt='-o', capsize=4, elinewidth=1.5,
                            color=style.color_comparison_primary,
                            label='σ>0 − σ=0')
                ax.plot(x_base, 0, 's', color='red', markersize=8, zorder=5,
                        label=f'基准 (σ={x_raw[baseline_idx]})')

                band_str = (f"${low:.2f}T < E < {high:.2f}T$"
                            if high else f"$E > {low:.2f}T$")
                ax.text(0.98, 0.95, band_str, transform=ax.transAxes,
                        ha='right', va='top', fontsize='medium', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                ax.axhline(0, color='black', linestyle='-', alpha=0.3, lw=1)
                ax.set_ylabel("Δ excess (%)")
                ax.legend(fontsize='x-small', loc='best', ncol=2)
                ax.grid(True, linestyle=':', alpha=0.4)

    # =========================================================================
    # 时序差分：target − baseline vs 时间
    # =========================================================================

    def _plot_timeseries_diff(self, run, target_series, baseline_series,
                              style, target_label, baseline_label):
        n = min(len(target_series), len(baseline_series))
        if n == 0:
            return

        singles = self._unpack_runs(run)
        dt = getattr(singles[0].sim, 'dt', 0.0) if singles else 0.0
        step_numbers = [_get_step_from_filename(f) or i
                        for i, f in enumerate(singles[0].particle_files[:n])]

        if dt > 0:
            x_time = np.array(step_numbers) * dt
            time_label = "时间 (s)"
        else:
            x_time = np.array(step_numbers, dtype=float)
            time_label = "时间步"

        is_grouped_t = isinstance(target_series[0], GroupedStepMetrics)
        is_grouped_b = isinstance(baseline_series[0], GroupedStepMetrics)

        with DataLayout(run, f"tail_ts_diff_{run.name}",
                        plot_ratio=(10, 3.5), ncols=2,
                        shared_xlabel=time_label) as layout:
            for low, high in self.INTERVALS:
                ax = layout.request_axes()
                label = f"{low:.2f}-{high:.2f}" if high else f"{low:.2f}-inf"

                t_vals = np.array([s.tail_excess.get(label, 0.0)
                                   for s in target_series[:n]])
                b_vals = np.array([s.tail_excess.get(label, 0.0)
                                   for s in baseline_series[:n]])
                diff = (t_vals - b_vals) * 100

                t_unc = np.array([s.tail_uncertainty.get(label, 0.0)
                                  for s in target_series[:n]])
                b_unc = np.array([s.tail_uncertainty.get(label, 0.0)
                                  for s in baseline_series[:n]])

                if is_grouped_t:
                    t_unc = np.sqrt(t_unc**2 + np.array([
                        s.tail_excess_std.get(label, 0.0)
                        for s in target_series[:n]])**2)
                if is_grouped_b:
                    b_unc = np.sqrt(b_unc**2 + np.array([
                        s.tail_excess_std.get(label, 0.0)
                        for s in baseline_series[:n]])**2)

                combined = np.sqrt(t_unc**2 + b_unc**2) * 100

                ax.fill_between(x_time, diff - combined, diff + combined,
                                color=style.color_comparison_primary, alpha=0.1,
                                label='合成误差')
                ax.plot(x_time, diff, color=style.color_comparison_primary,
                        lw=style.lw_base,
                        label=f'σ={target_label} − σ={baseline_label}')

                band_str = (f"${low:.2f}T < E < {high:.2f}T$"
                            if high else f"$E > {low:.2f}T$")
                ax.text(0.98, 0.95, band_str, transform=ax.transAxes,
                        ha='right', va='top', fontsize='medium', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                ax.axhline(0, color='black', linestyle='-', alpha=0.3, lw=0.8)
                ax.set_ylabel("Δ excess (%)")
                ax.legend(fontsize='x-small', loc='best', ncol=2)
                ax.grid(True, linestyle=':', alpha=0.4)
