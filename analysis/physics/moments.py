# analysis/physics/moments.py

from typing import NamedTuple

import numpy as np

from analysis.core.async_utils import asyncify
from analysis.core.cache import cached_op
from analysis.core.simulationSingle import SimulationRunSingle


class MomentsResult(NamedTuple):
    """加权能谱高阶矩：偏度、峰度、n阶中心矩。"""
    skewness: float      # 3阶标准化矩（加权）
    kurtosis: float      # 4阶超额标准化矩（加权, fisher=True, Gaussian=0）
    moment_3: float      # 3阶加权中心矩（原始值）
    moment_4: float      # 4阶加权中心矩（原始值）

    @staticmethod
    def null():
        return MomentsResult(0.0, 0.0, 0.0, 0.0)


@cached_op(file_dep="singleFile")
def compute_run_moments(
        run: 'SimulationRunSingle',
        fpath: str,
) -> MomentsResult:
    """
    计算粒子能谱的加权高阶统计矩。

    参数:
        run: 单次模拟数据对象
        fpath: 该步对应的粒子文件路径
    """
    spec = run.get_spectrum_from_path(fpath)

    if spec is None or spec.weights.size == 0:
        return MomentsResult.null()

    # 过滤零/负能量
    valid = spec.energies_MeV > 0
    E = spec.energies_MeV[valid]
    W = spec.weights[valid]

    if E.size < 3:
        return MomentsResult.null()

    w_sum = np.sum(W)
    if w_sum <= 0:
        return MomentsResult.null()

    # 加权均值
    mu = np.sum(W * E) / w_sum
    # 加权中心矩
    d = E - mu
    m2 = np.sum(W * d ** 2) / w_sum
    m3 = np.sum(W * d ** 3) / w_sum
    m4 = np.sum(W * d ** 4) / w_sum

    if m2 <= 0:
        return MomentsResult(0.0, 0.0, m3, m4)

    skewness = m3 / m2 ** 1.5
    excess_kurtosis = m4 / m2 ** 2 - 3.0

    return MomentsResult(
        skewness=float(skewness),
        kurtosis=float(excess_kurtosis),
        moment_3=float(m3),
        moment_4=float(m4),
    )


async_compute_moments = asyncify(compute_run_moments)
