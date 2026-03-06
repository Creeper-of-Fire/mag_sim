# analysis/modules/utils/physics_mj.py

"""
通用物理计算模块，专注于麦克斯韦-朱特纳 (Maxwell-Jüttner) 分布相关函数。
"""
import warnings
from typing import Optional

import numpy as np
from scipy.constants import k as kB, c, m_e, e
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import root_scalar
from scipy.special import kn as bessel_k
from scipy.special import kve as bessel_kve

# --- 物理常量 (可供其他模块导入) ---
ME_C2_J = m_e * c ** 2
"""电子静止能量 (焦耳)"""

J_PER_MEV = e * 1e6
"""1 MeV 等于多少焦耳"""

J_TO_KEV = 1.0 / (e * 1e3)
"""1 焦耳等于多少 keV"""

# --- 核心物理函数 ---

def solve_classical_temperature_kev(avg_ek_mev: float | np.floating) -> float:
    """
    根据平均动能反推经典 Maxwell-Boltzmann 温度 (keV)。
    此为非相对论近似，适用于 T << 511 keV 的情况。
    公式: <E_k> = 3/2 kT
    """
    if avg_ek_mev <= 0:
        return 0.0
    avg_ek_j = avg_ek_mev * J_PER_MEV
    # T_K = (2/3) * <E_k> / kB
    T_K = (2.0 / 3.0) * avg_ek_j / kB
    return T_K * kB * J_TO_KEV


def solve_mj_temperature_kev(avg_ek_mev: float | np.floating, guess_T_keV: Optional[float] = None) -> float:
    """
    根据平均动能反推相对论 Maxwell-Jüttner 温度 (keV)。
    通过数值求解 <E_k> = mc^2 * ( 3*theta + K1(1/theta)/K2(1/theta) - 1 )
    其中 theta = kT / mc^2。

    Args:
        avg_ek_mev: 平均动能 (MeV)。
        guess_T_keV: 求解器初值 (keV)。若为 None，则使用非相对论结果作为初值。

    Returns:
        拟合的温度 (keV)。
    """
    if avg_ek_mev <= 0:
        return 0.0

    target_avg_ek_j = avg_ek_mev * J_PER_MEV

    def mj_avg_energy(T_K: float) -> float:
        """给定温度(K)，计算 M-J 分布的平均动能(J)。"""
        if T_K <= 0:
            return -1.0  # 返回一个无效值以便求解器识别
        theta = (kB * T_K) / ME_C2_J
        # 避免在极低温度下出现数值问题
        if theta < 1e-9:
             return 1.5 * kB * T_K # 在数值极限下退化为经典情况

        # 使用 kve 代替 bessel_k，彻底消除大参数下溢！
        # K1(z)/K2(z) == bessel_kve(1, z)/bessel_kve(2, z)
        z = 1.0 / theta
        k_ratio = bessel_kve(1, z) / bessel_kve(2, z)

        return ME_C2_J * (3 * theta + k_ratio - 1.0)

    # 使用非相对论结果作为智能初值
    if guess_T_keV is None:
        T_guess_K = (2.0 / 3.0) * target_avg_ek_j / kB
    else:
        T_guess_K = (guess_T_keV / J_TO_KEV) / kB

    try:
        sol = root_scalar(
            lambda t: mj_avg_energy(t) - target_avg_ek_j,
            x0=T_guess_K,
            bracket=[T_guess_K * 0.01, T_guess_K * 100.0], # 扩大搜索范围
            method='brentq'
        )
        # 将结果从开尔文转回 keV
        return (sol.root * kB) * J_TO_KEV
    except ValueError:
        # 如果初值或区间有问题，返回0
        return 0.0


def calculate_mj_pdf(E_MeV: np.ndarray, T_keV: float) -> np.ndarray:
    """
    计算给定温度下 Maxwell-Jüttner 概率密度函数 f(E) 的值。
    函数已归一化，使得 ∫f(E)dE = 1。

    Args:
        E_MeV: 能量数组 (MeV)。
        T_keV: 等效温度 (keV)。

    Returns:
        每个能量点对应的概率密度 (单位: /MeV)。
    """
    if T_keV <= 0:
        return np.zeros_like(E_MeV)

    T_J = T_keV / J_TO_KEV
    theta = T_J / ME_C2_J
    z = 1.0 / theta

    # 使用 kve 替代 K_n 避免分母在低能时 underflow
    # 归一化系数 Z 的缩放版本 (不包含 e^{-z} 项)
    norm_scaled = 1.0 / (ME_C2_J * theta * bessel_kve(2, z))

    E_J = E_MeV * J_PER_MEV
    gamma = 1.0 + E_J / ME_C2_J
    # pc = sqrt( E_k^2 + 2*E_k*m*c^2 )
    pc_J = np.sqrt(E_J * (E_J + 2 * ME_C2_J))

    # 关键的物理对消：
    # exp(-gamma/theta) / K2(1/theta)
    # = exp(-gamma/theta) / ( kve * exp(-1/theta) )
    # = exp(-(gamma - 1)/theta) / kve
    # 注意 (gamma - 1)/theta 实际上就是 E_J / (k_B T)
    # 因此，我们把式子里面的-gamma / theta替换成-(gamma - 1.0) / theta，这里就相当于消掉了公式上面的e^静质量，而这个项就是bessel_kve2和K2的区别所在。

    # PDF (per Joule)
    pdf_per_joule = norm_scaled * (pc_J / ME_C2_J) * gamma * np.exp(-(gamma - 1.0) / theta)

    # 转换单位为 per MeV
    pdf_per_mev = pdf_per_joule * J_PER_MEV
    return pdf_per_mev


def calculate_mj_cdf(E_MeV: np.ndarray, T_keV: float) -> np.ndarray:
    """
    通过对 PDF进行数值积分，计算 Maxwell-Jüttner 分布的累积分布函数 F(E)。
    F(E) = ∫_0^E f(e) de

    Args:
        E_MeV: 能量数组 (MeV)，即积分的上限。
        T_keV: 等效温度 (keV)。

    Returns:
        每个能量点对应的累积概率。
    """
    if T_keV <= 0:
        return np.zeros_like(E_MeV)

    # 忽略在低能区可能出现的数值积分警告
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=IntegrationWarning,
            message="The integral is probably divergent, or slowly convergent."
        )

        # 使用 lambda 函数捕获 T_keV
        pdf_func = lambda e: calculate_mj_pdf(e, T_keV)

        # 对每个能量上限 E 进行积分
        cdf_values = np.array([quad(pdf_func, 0, e_max)[0] for e_max in E_MeV])

    # 钳制结果在 [0, 1] 范围内，防止微小的数值误差
    return np.clip(cdf_values, 0, 1)