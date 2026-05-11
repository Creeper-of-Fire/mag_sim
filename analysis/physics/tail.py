# analysis/physics/tail.py

import warnings
from typing import Optional, NamedTuple

import numpy as np
from scipy.integrate import quad, IntegrationWarning

from analysis.core.async_utils import asyncify
from analysis.core.cache import cached_op
from analysis.core.simulationSingle import SimulationRunSingle
from analysis.core.utils import console
from analysis.modules.utils import physics_mj
from analysis.physics.temperature import TemperatureResult


class TailResult(NamedTuple):
    """特定能段的尾部超额指标"""
    excess_ratio: float
    propagated_uncertainty: float
    threshold_low_MeV: float
    threshold_high_MeV: float

    @staticmethod
    def null():
        return TailResult(
            excess_ratio=0.0,
            propagated_uncertainty=0.0,
            threshold_low_MeV=0.0,
            threshold_high_MeV=0.0
        )


@cached_op(file_dep="auto")
def compute_run_tail_metrics(
        run: 'SimulationRunSingle',
        temperature_metrics: TemperatureResult,
        f_low: float,
        f_high: Optional[float] = None,
        fpath: str = None,
) -> TailResult:
    """
    计算单个 Run 在指定能量区间 [f_low*T, f_high*T] 内的物理度量。并执行极其严谨的统计学误差传递。

    参数:
        run: 单次模拟数据对象
        f_low: 区间下限倍数 (E > f_low * T)
        f_high: 区间上限倍数 (E < f_high * T)，若为 None 则表示到正无穷
        fpath: 该步对应的粒子文件路径（用于缓存绑定）
    """
    spec = run.get_spectrum_from_path(fpath)

    T_keV = temperature_metrics.T_keV
    sigma_T = temperature_metrics.sigma_T
    total_weight = temperature_metrics.total_weight
    total_energy_MeV = temperature_metrics.total_energy_MeV
    max_sim_energy_MeV = temperature_metrics.max_energy_MeV

    if T_keV <= 0 or total_energy_MeV <= 0:
        return TailResult.null()

    # =========================================================================
    # 3. 步骤三：定义理论尾部能量函数，并计算其误差 sigma_Eth
    # =========================================================================
    def compute_theoretical_tail_energy(T_val: float) -> float:
        e_low = (f_low * T_val) / 1000.0
        # 如果没有上限，或者上限超过了模拟最大能量，则积分到最大能量
        if f_high is None:
            e_high = max_sim_energy_MeV
        else:
            e_high = (f_high * T_val) / 1000.0

        def integrand(e):
            return e * physics_mj.calculate_mj_pdf(np.array([e]), T_val)[0]

        def pdf_func(e):
            return physics_mj.calculate_mj_pdf(np.array([e]), T_val)[0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            quad_result = quad(integrand, e_low, e_high, limit=200)[0]
            prob_norm = quad(pdf_func, 0, max_sim_energy_MeV, limit=200)[0]

        if prob_norm <= 0: return 0.0
        return (quad_result / prob_norm) * total_weight

    # 基准理论能量与由于温度波动引起的理论误差
    th_tail_energy_MeV = compute_theoretical_tail_energy(T_keV)

    # 数值求导 |dE_th / dT|
    delta_T = T_keV * 1e-4
    th_tail_plus = compute_theoretical_tail_energy(T_keV + delta_T)
    th_tail_minus = compute_theoretical_tail_energy(T_keV - delta_T)
    dEth_dT = (th_tail_plus - th_tail_minus) / (2 * delta_T)

    sigma_th_tail_energy = abs(dEth_dT) * sigma_T

    # =========================================================================
    # 4. 步骤四：计算模拟尾部的泊松/散粒散布误差 sigma_Esim
    # =========================================================================
    e_low_th = (f_low * T_keV) / 1000.0
    e_high_th = (f_high * T_keV / 1000.0) if f_high else float('inf')

    mask = (spec.energies_MeV >= e_low_th) & (spec.energies_MeV < e_high_th)

    if not np.any(mask):
        sim_tail_energy_MeV = 0.0
        sigma_sim_tail_energy = 0.0
    else:
        sim_tail_energy_MeV = np.sum(spec.energies_MeV[mask] * spec.weights[mask])
        # 核心：加权蒙特卡洛计数的方差公式 Var = sum( (W_i * E_i)^2 )
        var_sim_tail_energy = np.sum((spec.energies_MeV[mask] * spec.weights[mask]) ** 2)
        sigma_sim_tail_energy = np.sqrt(var_sim_tail_energy)

    # =========================================================================
    # 5. 步骤五：合成最终的理论 Excess 误差
    # =========================================================================
    excess_energy_MeV = sim_tail_energy_MeV - th_tail_energy_MeV
    excess_ratio = excess_energy_MeV / total_energy_MeV

    # 独立误差平方和开根号
    sigma_excess_energy = np.sqrt(sigma_sim_tail_energy ** 2 + sigma_th_tail_energy ** 2)
    # 转换为占比的相对误差
    propagated_uncertainty = sigma_excess_energy / total_energy_MeV

    return TailResult(
        excess_ratio=excess_ratio,
        propagated_uncertainty=propagated_uncertainty,  # <--- 这是我们手算出来的终极理论误差！
        threshold_low_MeV=e_low_th,
        threshold_high_MeV=e_high_th if f_high else max_sim_energy_MeV
    )


async_compute_tail = asyncify(compute_run_tail_metrics)
