# analysis/physics/temperature.py

from typing import Any, NamedTuple

import numpy as np

from analysis.core.async_utils import asyncify
from analysis.core.cache import cached_op
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.modules.utils import physics_mj


class TemperatureResult(NamedTuple):
    """光谱基础物理量及统计误差"""
    T_keV: float
    sigma_T: float
    total_weight: float
    total_energy_MeV: float
    avg_energy_MeV: float
    max_energy_MeV: float

    @staticmethod
    def null():
        return TemperatureResult(
            T_keV=0.0, sigma_T=0.0, total_weight=0.0,
            total_energy_MeV=0.0, avg_energy_MeV=0.0, max_energy_MeV=0.0,
        )


@cached_op(file_dep="singleFile")
def compute_run_temperature_metrics(
        run: 'SimulationRunSingle',
        fpath: str,
) -> TemperatureResult:
    """
    计算单个 Run 的温度和温度误差。并执行极其严谨的统计学误差传递。

    参数:
        run: 单次模拟数据对象
        fpath: 该步对应的粒子文件路径（用于缓存绑定）
    """
    spec = run.get_spectrum_from_path(fpath)

    if spec is None or spec.weights.size == 0:
        return TemperatureResult.null()

    # =========================================================================
    # 0. 精确基础统计与有效粒子数 (Effective Sample Size)
    # =========================================================================
    total_energy_MeV: Any = np.sum(spec.energies_MeV * spec.weights)
    total_weight = np.sum(spec.weights)

    if total_weight == 0 or total_energy_MeV <= 0:
        return TemperatureResult.null()

    avg_energy_MeV = total_energy_MeV / total_weight
    max_energy_MeV = np.max(spec.energies_MeV)

    # PIC 加权统计中的关键：有效粒子数 N_eff
    V1 = total_weight
    V2 = np.sum(spec.weights ** 2)
    if V2 > 0:
        N_eff = (V1 ** 2) / V2
    else:
        N_eff = 1.0

    # =========================================================================
    # 1. 步骤一：计算平均能量的标准误 sigma_<E>
    # =========================================================================
    # 加权能量方差 Var(E)
    var_E = np.sum(spec.weights * (spec.energies_MeV - avg_energy_MeV) ** 2) / V1
    # 平均值的标准误
    sigma_avg_E = np.sqrt(var_E / N_eff)

    # =========================================================================
    # 2. 步骤二：误差传递到 MJ 温度 sigma_T
    # =========================================================================
    T_keV = physics_mj.solve_mj_temperature_kev(avg_energy_MeV)

    # 使用数值微积分 (中心差分法) 计算导数 |dT / d<E>|
    delta_E = avg_energy_MeV * 1e-4  # 取 0.01% 作为微小扰动
    T_plus = physics_mj.solve_mj_temperature_kev(avg_energy_MeV + delta_E, guess_T_keV=T_keV)
    T_minus = physics_mj.solve_mj_temperature_kev(avg_energy_MeV - delta_E, guess_T_keV=T_keV)

    dT_dE = (T_plus - T_minus) / (2 * delta_E)
    sigma_T = abs(dT_dE) * sigma_avg_E

    return TemperatureResult(
        T_keV=T_keV,
        sigma_T=sigma_T,
        total_weight=total_weight,
        total_energy_MeV=total_energy_MeV,
        avg_energy_MeV=avg_energy_MeV,
        max_energy_MeV=max_energy_MeV
    )


async_compute_temperature = asyncify(compute_run_temperature_metrics)
