# analysis/modules/utils/comparison_utils.py
from typing import List, Tuple

import numpy as np

from analysis.core.simulation import SimulationRun


def create_common_energy_bins(
        runs: List[SimulationRun],
        num_bins: int = 200,
        log_scale: bool = True,
        padding_factor_min: float = 0.9,
        padding_factor_max: float = 1.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为一系列模拟运行创建统一的、共享的能量分箱。

    这个函数会扫描所有给定 run 的初始和最终能谱，找到全局的能量范围，
    然后创建一个适用于所有 run 对比的能量分箱。

    Args:
        runs: 待分析的 SimulationRun 对象列表。
        num_bins: 分箱数量。
        log_scale: 是否使用对数分箱。
        padding_factor_min: 最小能量的填充因子。
        padding_factor_max: 最大能量的填充因子。

    Returns:
        A tuple of (bins, centers, widths).

    Raises:
        ValueError: 如果所有 run 中都没有有效的能谱数据。
    """
    all_energies = []
    for run in runs:
        if run.initial_spectrum and run.initial_spectrum.energies_MeV.size > 0:
            all_energies.append(run.initial_spectrum.energies_MeV)
        if run.final_spectrum and run.final_spectrum.energies_MeV.size > 0:
            all_energies.append(run.final_spectrum.energies_MeV)

    if not all_energies:
        raise ValueError("在所有提供的模拟中都没有找到有效的能谱数据。")

    combined = np.concatenate(all_energies)
    positive = combined[combined > 0]

    if positive.size < 2:
        raise ValueError("有效的正能量数据点过少，无法创建分箱。")

    # 全局能量范围
    min_e = max(positive.min() * padding_factor_min, 1e-4)
    max_e = positive.max() * padding_factor_max

    if log_scale:
        bins = np.logspace(np.log10(min_e), np.log10(max_e), num_bins)
    else:
        bins = np.linspace(min_e, max_e, num_bins)

    centers = np.sqrt(bins[:-1] * bins[1:]) if log_scale else (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)

    return bins, centers, widths