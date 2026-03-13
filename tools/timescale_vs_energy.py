# analysis/modules/timescale_vs_energy.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, epsilon_0, m_e, mu_0,k
from scipy.constants import sigma as SIGMA_SB  # Stefan-Boltzmann constant

# 尝试导入字体设置工具
try:
    from analysis.utils import setup_chinese_font

    setup_chinese_font()
except ImportError:
    print("[警告] 未找到字体设置工具，中文可能无法正常显示。")


class TimescaleCalculator:
    """
    计算不同能量（洛伦兹因子gamma）下，单个粒子的各种物理特征时间。
    """

    def __init__(self, n_e: float, T_K: float, B_field: float):
        """
        初始化背景等离子体环境。

        Args:
            n_e (float): 背景电子数密度 (m^-3)
            T_K (float): 背景光子温度 (Kelvin), 用于计算IC冷却
            B_field (float): 背景磁场强度 (Tesla), 用于计算加速和回旋
        """
        self.n_e = n_e
        self.T_K = T_K
        self.B_field = B_field
        self.coulomb_log = 15  # 库仑对数，通常取10-20

    def get_collision_time(self, gamma: np.ndarray) -> np.ndarray:
        """
        计算相对论修正的碰撞时间 (s)。
        tau_coll ~ gamma^2 * v^3。
        """
        v = c * np.sqrt(1 - 1 / gamma**2)
        v[v == 0] = 1e-6
        # 碰撞频率 nu_coll ~ 1 / (gamma^2 * v^3)
        nu_coll = (self.n_e * e**4 * self.coulomb_log) / (4 * np.pi * epsilon_0**2 * m_e**2 * gamma**2 * v**3)
        return 1.0 / nu_coll

    def get_cooling_time_ic(self, gamma: np.ndarray) -> np.ndarray:
        """
        计算逆康普顿 (IC) 冷却时间 (s)。
        tau_cool 随能量增加而减少 (高能粒子冷却极快)。
        """
        # 背景光子能量密度 U_ph = a * T^4 = (4*sigma_sb/c) * T^4
        U_rad = (4 * SIGMA_SB / c) * self.T_K ** 4
        sigma_T = 6.6524e-29  # 汤姆逊散射截面

        # 冷却功率 P_cool = (4/3) * sigma_T * c * U_rad * (v/c)^2 * gamma^2
        # 近似 (v/c)^2 ~ 1 for relativistic particles
        power = (4 / 3) * sigma_T * c * U_rad * gamma ** 2

        # 粒子总能量 E_k = (gamma - 1) * m_e * c^2
        total_energy = (gamma - 1) * m_e * c ** 2

        # tau_cool = E_k / P_cool
        tau = total_energy / power
        tau[power == 0] = np.inf
        return tau

    def get_acceleration_time_rec(self, gamma: np.ndarray) -> np.ndarray:
        """
        估算磁重联加速时间 (s)。
        假设一个经典的加速率 E_rec ~ 0.1 * vA * B。
        """
        if self.B_field == 0:
            return np.full_like(gamma, np.inf)

        # 估算 Alfven 速度 (这里用光速作为上限)
        # 真实的 vA 需要考虑等离子体总能量密度，但这里 c 是一个不错的近似
        v_A = c

        # 加速电场
        E_rec = 0.1 * v_A * self.B_field

        # 加速功率 P_acc = F * v ~ (q * E_rec) * c
        power = e * E_rec * c

        total_energy = (gamma - 1) * m_e * c ** 2

        # tau_accel = E_k / P_acc
        tau = total_energy / power
        tau[power == 0] = np.inf
        return tau

    def get_gyration_time(self, gamma: np.ndarray) -> np.ndarray:
        """计算回旋周期 (s)"""
        if self.B_field == 0:
            return np.full_like(gamma, np.inf)

        omega_ce = (e * self.B_field) / (gamma * m_e)
        return 2 * np.pi / omega_ce


def plot_timescales(scenario_name: str, calculator: TimescaleCalculator, E_kin_eV: np.ndarray):
    """
    绘制 timescale vs energy 的核心图像。
    """
    gamma = 1 + E_kin_eV * e / (m_e * c ** 2)

    # 计算所有时间尺度
    t_coll = calculator.get_collision_time(gamma)
    t_accel = calculator.get_acceleration_time_rec(gamma)
    t_gyro = calculator.get_gyration_time(gamma)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘图
    ax.loglog(E_kin_eV / 1e3, t_coll, label=r"碰撞时间 $\tau_{coll} \propto \gamma^2$", lw=2.5, color='orange')
    ax.loglog(E_kin_eV / 1e3, t_accel, label=r"加速时间 $\tau_{acc} \propto \gamma$", lw=2.5, color='green')
    ax.loglog(E_kin_eV / 1e3, t_gyro, label=r"回旋周期 $\tau_{gyro} \propto \gamma$", lw=2, color='blue', ls='--')

    # 找到关键交叉点
    # 加速 vs 碰撞 (非热阈值)
    idx_acc_coll = np.argwhere(np.diff(np.sign(t_accel - t_coll))).flatten()
    if len(idx_acc_coll) > 0:
        cross_E1 = E_kin_eV[idx_acc_coll[0]] / 1e3
        ax.axvline(cross_E1, color='gray', ls=':', lw=2)
        ax.text(cross_E1 * 1.1, ax.get_ylim()[0] * 1.5, '非热加速阈值\n($\\tau_{acc} < \\tau_{coll}$)',
                rotation=90, va='bottom', ha='left', color='gray', fontsize=10)

    # 标注区域
    y_range = ax.get_ylim()
    ax.fill_between([10, 84 * 3], y_range[0], y_range[1], color='gray', alpha=0.1, label='热粒子区域')

    # 美化
    ax.set_xlabel("粒子动能 (keV)", fontsize=14)
    ax.set_ylabel("特征时间 (秒)", fontsize=14)
    ax.set_title(f"物理过程特征时间 vs 能量\n({scenario_name})", fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)
    ax.set_ylim(bottom=min(t_gyro.min(), t_accel.min()) * 0.1)  # 调整Y轴范围

    plt.tight_layout()
    plt.savefig(f"timescales_{scenario_name.replace(' ', '_')}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # --- 配置你的物理场景 ---
    # 使用你熟悉的“氘瓶颈”参数
    scenario = {
        "name": "Deuterium Bottleneck (sigma=0.05)",
        "n_e": 7.28e33,  # 电子数密度 (m^-3)
        "T_eV": 84480.0,  # 背景温度 (eV)
        "sigma": 0.05  # 假设的磁化参数
    }

    # 从 sigma 反推磁场 B
    T_J = scenario["T_eV"] * e
    # 假设正负电子对等离子体，总能量密度
    U_p = 2 * scenario["n_e"] * (m_e * c ** 2 + 3 * T_J)
    B_field = np.sqrt(scenario["sigma"] * 2 * mu_0 * U_p)
    print(f"场景: {scenario['name']}")
    print(f"  - 背景温度: {scenario['T_eV'] / 1e3:.1f} keV")
    print(f"  - 推算磁场: {B_field:.2f} T")

    # 初始化计算器
    T_K = T_J / k
    calculator = TimescaleCalculator(n_e=scenario["n_e"], T_K=T_K, B_field=B_field)

    # 定义我们关心的能量范围 (从热能到极高能)
    # 比如从 10 keV 到 10 GeV
    E_kin_eV = np.logspace(np.log10(10e3), np.log10(10e9), 200)

    # 生成并显示图像
    plot_timescales(scenario["name"], calculator, E_kin_eV)