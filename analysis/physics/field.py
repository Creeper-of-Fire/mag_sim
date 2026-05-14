# analysis/physics/field.py

import logging
from typing import Tuple

import numpy as np
from scipy.constants import mu_0, e, epsilon_0

from analysis.core.cache import cached_op
from analysis.core.data_loader import _get_step_from_filename, h5open
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.physics.temperature import compute_run_temperature_metrics


@cached_op(file_dep="singleFile")
def get_mean_u_mag(run: 'SimulationRunSingle', fpath: str) -> float:
    """计算单个 HDF5 场文件中的全空间平均磁场能量密度 (J/m³)。"""
    step = _get_step_from_filename(fpath)
    with h5open(fpath, 'r') as f:
        bp = f"/data/{step}/fields/B"
        b_sq_mean = np.mean(f[bp + '/x'][:] ** 2 + f[bp + '/y'][:] ** 2 + f[bp + '/z'][:] ** 2)
    return float(b_sq_mean / (2 * mu_0))


@cached_op(file_dep="singleFile")
def get_mean_u_elec(run: 'SimulationRunSingle', fpath: str) -> float:
    """计算全空间平均电场能量密度 (0.5 * epsilon_0 * E²)。"""
    step = _get_step_from_filename(fpath)
    with h5open(fpath, 'r') as f:
        ep = f"/data/{step}/fields/E"
        e_sq_mean = np.mean(f[ep + '/x'][:] ** 2 + f[ep + '/y'][:] ** 2 + f[ep + '/z'][:] ** 2)
    return float(0.5 * epsilon_0 * e_sq_mean)


def compute_run_energy_partition(run: 'SimulationRunSingle', step_index: int) -> Tuple[float, float, float]:
    """返回 (磁能占比, 电能占比, 总场能占比)"""
    fpath = run.get_particle_file(step_index)
    t_metrics = compute_run_temperature_metrics(run, fpath=fpath)
    if t_metrics.avg_energy_MeV <= 0:
        return np.nan, np.nan, np.nan

    u_kin = (2.0 * run.sim.n_plasma) * (t_metrics.avg_energy_MeV * e * 1e6)
    try:
        field_fpath = run.get_field_file(step_index)
    except IndexError:
        logging.debug(f"compute_run_energy_partition: step_index={step_index} 无场文件")
        return np.nan, np.nan, np.nan

    u_mag = get_mean_u_mag(run, field_fpath)
    u_elec = get_mean_u_elec(run, field_fpath)
    u_total = u_kin + u_mag + u_elec

    return float(u_mag / u_total), float(u_elec / u_total), float((u_mag + u_elec) / u_total)


def compute_run_energy_densities_normalized(run: 'SimulationRunSingle', step_index: int) -> Tuple[float, float, float, float]:
    """返回归一化能量密度 (以 n₀·mₑc² 为单位)"""
    fpath = run.get_particle_file(step_index)
    t_metrics = compute_run_temperature_metrics(run, fpath=fpath)
    if t_metrics.avg_energy_MeV <= 0:
        return 0.0, 0.0, 0.0, 0.0

    m_e_c2_J = 8.1871e-14
    n0 = run.sim.n_plasma
    norm_factor = n0 * m_e_c2_J

    u_kin_J = (2.0 * n0) * (t_metrics.avg_energy_MeV * e * 1e6)
    u_kin_norm = u_kin_J / norm_factor

    try:
        field_fpath = run.get_field_file(step_index)
    except IndexError:
        logging.debug(f"compute_run_energy_densities_normalized: step_index={step_index} 无场文件")
        return u_kin_norm, 0.0, 0.0, u_kin_norm

    u_mag_norm = get_mean_u_mag(run, field_fpath) / norm_factor
    u_elec_norm = get_mean_u_elec(run, field_fpath) / norm_factor
    u_total_norm = u_kin_norm + u_mag_norm + u_elec_norm

    return u_kin_norm, u_mag_norm, u_elec_norm, u_total_norm


# ---------------------------------------------------------------------------
# Async 导出（per-function 专属线程池）
# ---------------------------------------------------------------------------

from analysis.core.async_utils import asyncify

async_compute_energy_partition = asyncify(compute_run_energy_partition)
async_compute_energy_densities_normalized = asyncify(compute_run_energy_densities_normalized)
