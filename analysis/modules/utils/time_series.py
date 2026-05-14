# analysis/modules/utils/time_series.py
"""
Pipeline utilities for time-series extraction and temporal aggregation.

Stage 1: extract_tail_time_series_async — per-step metrics with per-step parallelism
Stage 2: avg_last_n / avg_all — pluggable aggregation strategies
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, NamedTuple, Tuple

from analysis.core.async_utils import gather_dict
from analysis.core.cache import cached_op
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.physics.temperature import (
    TemperatureResult, async_compute_temperature,
)
from analysis.physics.tail import async_compute_tail
from analysis.physics.field import (
    async_compute_energy_partition, async_compute_energy_densities_normalized,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class StepMetrics(NamedTuple):
    """Single time-step metrics from tail_statisticsV2."""
    T_keV: float
    sigma_T: float
    total_energy_MeV: float
    tail_excess: Dict[str, float]  # keyed by interval label e.g. "1.00-2.00"
    tail_uncertainty: Dict[str, float]
    mag_fraction: float
    elec_fraction: float
    field_fraction: float
    kin_density_norm: float
    mag_density_norm: float
    elec_density_norm: float
    total_density_norm: float


class AggregatedMetrics(NamedTuple):
    """Result of temporal aggregation over a StepMetrics series."""
    T_keV: float
    sigma_T: float
    total_energy_MeV: float
    tail_excess: Dict[str, float]
    tail_uncertainty: Dict[str, float]
    mag_fraction: float
    elec_fraction: float
    field_fraction: float
    n_steps_averaged: int
    kin_density_norm: float
    mag_density_norm: float
    elec_density_norm: float
    total_density_norm: float
    # 跨 run 统计误差 (仅 Group 时有值，单 run 时为 0)
    T_keV_std: float = 0.0
    tail_excess_std: Dict[str, float] = {}
    mag_fraction_std: float = 0.0
    elec_fraction_std: float = 0.0
    field_fraction_std: float = 0.0
    kin_density_norm_std: float = 0.0
    mag_density_norm_std: float = 0.0
    elec_density_norm_std: float = 0.0
    total_density_norm_std: float = 0.0


class GroupedStepMetrics(NamedTuple):
    """Time-step metrics averaged across multiple runs (mean + cross-run std)."""
    T_keV: float;  T_keV_std: float
    tail_excess: Dict[str, float];  tail_excess_std: Dict[str, float]
    tail_uncertainty: Dict[str, float]  # 理论传递误差均值
    mag_fraction: float;  mag_fraction_std: float
    elec_fraction: float;  elec_fraction_std: float
    field_fraction: float;  field_fraction_std: float
    kin_density_norm: float;  kin_density_norm_std: float
    mag_density_norm: float;  mag_density_norm_std: float
    elec_density_norm: float;  elec_density_norm_std: float
    total_density_norm: float;  total_density_norm_std: float


# ---------------------------------------------------------------------------
# Stage 1: Time-series extraction
# ---------------------------------------------------------------------------

@cached_op(file_dep="all")
async def extract_tail_time_series_async(
        run: SimulationRunSingle,
        intervals: List[Tuple[float, Optional[float]]],
        field_files_needed: bool = True,
) -> List[StepMetrics]:
    """
    Async extraction with per-step parallelism.

    All steps are submitted concurrently. Within each step, tail metrics for
    different intervals run in parallel after temperature completes.
    """
    n_steps = len(run.particle_files)
    if n_steps == 0:
        return []

    async def _step(step_idx: int) -> StepMetrics:
        fpath = run.get_particle_file(step_idx)

        t_metrics: TemperatureResult = await async_compute_temperature(run, fpath=fpath)

        tail_tasks = {}
        for low, high in intervals:
            label = f"{low:.2f}-{high:.2f}" if high else f"{low:.2f}-inf"
            tail_tasks[label] = async_compute_tail(
                run, temperature_metrics=t_metrics,
                f_low=low, f_high=high, fpath=fpath,
            )
        tail_results = await gather_dict(tail_tasks)

        tail_excess = {k: v.excess_ratio for k, v in tail_results.items()}
        tail_unc = {k: v.propagated_uncertainty for k, v in tail_results.items()}

        mag_f, elec_f, field_f = np.nan, np.nan, np.nan
        kin_d, mag_d, elec_d, tot_d = 0.0, 0.0, 0.0, 0.0
        if field_files_needed:
            mag_f, elec_f, field_f = await async_compute_energy_partition(run, step_idx)
            kin_d, mag_d, elec_d, tot_d = await async_compute_energy_densities_normalized(run, step_idx)

        return StepMetrics(
            T_keV=t_metrics.T_keV,
            sigma_T=t_metrics.sigma_T,
            total_energy_MeV=t_metrics.total_energy_MeV,
            tail_excess=tail_excess,
            tail_uncertainty=tail_unc,
            mag_fraction=mag_f,
            elec_fraction=elec_f,
            field_fraction=field_f,
            kin_density_norm=kin_d,
            mag_density_norm=mag_d,
            elec_density_norm=elec_d,
            total_density_norm=tot_d,
        )

    return list(await asyncio.gather(*[_step(i) for i in range(n_steps)]))


def _aggregate_grouped_series(all_series: List[List[StepMetrics]]) -> List[GroupedStepMetrics]:
    """Aggregate per-run time series into grouped metrics with mean and cross-run std."""
    n_steps = min(len(s) for s in all_series)
    labels = list(all_series[0][0].tail_excess.keys())

    result: List[GroupedStepMetrics] = []
    for step in range(n_steps):
        steps = [s[step] for s in all_series]

        def _mean_std(attr):
            vals = [getattr(s, attr) for s in steps]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                return 0.0, 0.0
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return m, s

        T_m, T_s = _mean_std('T_keV')
        mag_m, mag_s = _mean_std('mag_fraction')
        elec_m, elec_s = _mean_std('elec_fraction')
        field_m, field_s = _mean_std('field_fraction')
        kin_m, kin_s = _mean_std('kin_density_norm')
        magd_m, magd_s = _mean_std('mag_density_norm')
        elecd_m, elecd_s = _mean_std('elec_density_norm')
        totd_m, totd_s = _mean_std('total_density_norm')

        avg_excess: Dict[str, float] = {}
        std_excess: Dict[str, float] = {}
        avg_unc: Dict[str, float] = {}
        for label in labels:
            vals_e = [s.tail_excess.get(label, 0.0) for s in steps]
            avg_excess[label] = float(np.mean(vals_e))
            std_excess[label] = float(np.std(vals_e, ddof=1)) if len(vals_e) > 1 else 0.0
            vals_u = [s.tail_uncertainty.get(label, 0.0) for s in steps]
            avg_unc[label] = float(np.mean(vals_u))

        result.append(GroupedStepMetrics(
            T_keV=T_m, T_keV_std=T_s,
            tail_excess=avg_excess, tail_excess_std=std_excess,
            tail_uncertainty=avg_unc,
            mag_fraction=mag_m, mag_fraction_std=mag_s,
            elec_fraction=elec_m, elec_fraction_std=elec_s,
            field_fraction=field_m, field_fraction_std=field_s,
            kin_density_norm=kin_m, kin_density_norm_std=kin_s,
            mag_density_norm=magd_m, mag_density_norm_std=magd_s,
            elec_density_norm=elecd_m, elec_density_norm_std=elecd_s,
            total_density_norm=totd_m, total_density_norm_std=totd_s,
        ))

    return result

async def extract_grouped_time_series_async(
        group_runs: List[SimulationRunSingle],
        intervals: List[Tuple[float, Optional[float]]],
        field_files_needed: bool = True,
) -> List[GroupedStepMetrics]:
    """Extract all runs in parallel, then aggregate."""
    all_series = await asyncio.gather(*[
        extract_tail_time_series_async(r, intervals, field_files_needed)
        for r in group_runs
    ])
    all_series = [s for s in all_series if s]
    if not all_series:
        return []

    return _aggregate_grouped_series(all_series)


# ---------------------------------------------------------------------------
# Stage 2: Temporal aggregation strategies
# ---------------------------------------------------------------------------

def _avg_last_n_impl(series, n: int = 5) -> Optional[AggregatedMetrics]:
    """Pure aggregation: average the last *n* time steps. No I/O, no caching."""
    if not series:
        return None

    tail = series[-n:] if n < len(series) else series
    n_actual = len(tail)
    is_grouped = isinstance(tail[0], GroupedStepMetrics)

    labels = list(tail[0].tail_excess.keys())

    avg_T = float(np.mean([s.T_keV for s in tail]))
    avg_energy = 0.0
    avg_sigma_T = 0.0
    if not is_grouped:
        avg_sigma_T = float(np.sqrt(np.mean([s.sigma_T ** 2 for s in tail])))
        avg_energy = float(np.mean([s.total_energy_MeV for s in tail]))

    avg_tail: Dict[str, float] = {}
    avg_unc: Dict[str, float] = {}
    std_tail: Dict[str, float] = {}
    for label in labels:
        vals = [s.tail_excess[label] for s in tail if label in s.tail_excess]
        avg_tail[label] = float(np.mean(vals)) if vals else 0.0
        uncs = [s.tail_uncertainty[label] for s in tail if label in s.tail_uncertainty]
        avg_unc[label] = float(np.sqrt(np.mean([u ** 2 for u in uncs]))) if uncs else 0.0
        if is_grouped:
            stds = [s.tail_excess_std.get(label, 0.0) for s in tail]
            std_tail[label] = float(np.sqrt(np.mean([s ** 2 for s in stds])))

    avg_mag = float(np.nanmean([s.mag_fraction for s in tail]))
    avg_elec = float(np.nanmean([s.elec_fraction for s in tail]))
    avg_field = float(np.nanmean([s.field_fraction for s in tail]))
    avg_kin_d = float(np.nanmean([s.kin_density_norm for s in tail]))
    avg_mag_d = float(np.nanmean([s.mag_density_norm for s in tail]))
    avg_elec_d = float(np.nanmean([s.elec_density_norm for s in tail]))
    avg_tot_d = float(np.nanmean([s.total_density_norm for s in tail]))

    # 跨 run 统计误差（仅 grouped 有意义）
    kw = {}
    if is_grouped:
        kw = dict(
            T_keV_std=float(np.sqrt(np.mean([s.T_keV_std ** 2 for s in tail]))),
            tail_excess_std=std_tail,
            mag_fraction_std=float(np.sqrt(np.nanmean([s.mag_fraction_std ** 2 for s in tail]))),
            elec_fraction_std=float(np.sqrt(np.nanmean([s.elec_fraction_std ** 2 for s in tail]))),
            field_fraction_std=float(np.sqrt(np.nanmean([s.field_fraction_std ** 2 for s in tail]))),
            kin_density_norm_std=float(np.sqrt(np.nanmean([s.kin_density_norm_std ** 2 for s in tail]))),
            mag_density_norm_std=float(np.sqrt(np.nanmean([s.mag_density_norm_std ** 2 for s in tail]))),
            elec_density_norm_std=float(np.sqrt(np.nanmean([s.elec_density_norm_std ** 2 for s in tail]))),
            total_density_norm_std=float(np.sqrt(np.nanmean([s.total_density_norm_std ** 2 for s in tail]))),
        )

    return AggregatedMetrics(
        T_keV=avg_T,
        sigma_T=avg_sigma_T,
        total_energy_MeV=avg_energy,
        tail_excess=avg_tail,
        tail_uncertainty=avg_unc,
        mag_fraction=avg_mag,
        elec_fraction=avg_elec,
        field_fraction=avg_field,
        n_steps_averaged=n_actual,
        kin_density_norm=avg_kin_d,
        mag_density_norm=avg_mag_d,
        elec_density_norm=avg_elec_d,
        total_density_norm=avg_tot_d,
        **kw,
    )


@cached_op(file_dep="all")
async def compute_run_avg_last_n(
        run: SimulationRunSingle,
        intervals: List[Tuple[float, Optional[float]]],
        n: int = 5,
        field_files_needed: bool = True,
) -> Optional[AggregatedMetrics]:
    """Per-run cached: extract time series + aggregate last N steps."""
    series = await extract_tail_time_series_async(run, intervals, field_files_needed)
    return _avg_last_n_impl(series, n) if series else None


def avg_last_n(series, n: int = 5) -> Optional[AggregatedMetrics]:
    """Aggregate last N steps of already-computed series (for grouped results)."""
    return _avg_last_n_impl(series, n)
