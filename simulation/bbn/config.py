#!/usr/bin/env python3
"""BBN 混合 PIC 模拟参数配置。

所有物理参数从 CosmoCons.dat 数据源自动计算，用户只需指定：
- 温度点（CosmoCons 行号）
- 模拟网格/步数等数值参数
"""

from pathlib import Path

import numpy as np
from scipy.constants import e as q_e, c, epsilon_0, m_e, m_p

# CosmoCons.dat 本地路径
COSMOCONS_PATH = Path(__file__).resolve().parents[2] / "tools" / "bbn_physics" / "CosmoCons.dat"

# CosmoCons.dat 14 列定义
_CC_COLUMNS = [
    "Temp",      # MeV
    "time",      # s
    "rhog",      # g/cm^3
    "rhoe",      # g/cm^3
    "rhone",     # g/cm^3
    "rhob",      # g/cm^3
    "phie",      # dimensionless
    "rhotot",    # g/cm^3
    "H",         # 1/s
    "Ne",        # 1/cm^3
    "Ng",        # 1/cm^3
    "nLamdg",
    "LamdD",
    "c_over_H",  # cm
]


def load_cosmocons_row(row_index: int) -> dict:
    """从 CosmoCons.dat 加载指定行，返回 CGS 物理量字典。"""
    lines = COSMOCONS_PATH.read_text().strip().split("\n")
    data_lines = [l for l in lines
                  if l.strip() and not l.strip().startswith("Temp") and "---" not in l]

    row = data_lines[row_index]
    vals = row.split()
    if len(vals) != 14:
        raise ValueError(f"CosmoCons 行 {row_index} 应有 14 列, 实际 {len(vals)}")

    result = {}
    for name, val_str in zip(_CC_COLUMNS, vals):
        result[name] = np.inf if val_str == "INF" else float(val_str)
    return result


def _cgs_to_si(row: dict) -> dict:
    """将 CosmoCons CGS 数据转为 SI。"""
    return {
        "T_eV":       row["Temp"] * 1e6,              # MeV -> eV
        "T_MeV":      row["Temp"],
        "time_s":     row["time"],
        "n_e":        row["Ne"] * 1e6,                 # cm^-3 -> m^-3
        "n_gamma":    row["Ng"] * 1e6,
        "rho_b":      row["rhob"] * 1e3,               # g/cm^3 -> kg/m^3
        "rho_g":      row["rhog"] * 1e3,
        "rho_e":      row["rhoe"] * 1e3,
        "rho_nu":     row["rhone"] * 1e3,
        "rho_tot":    row["rhotot"] * 1e3,
        "H":          row["H"],
        "phie":       row["phie"],
        "c_over_H":   row["c_over_H"] * 1e-2,          # cm -> m
    }


def compute_plasma_params(n_e: float, n_ion: float, T_e_eV: float, ion_mass: float):
    """从基本物理量计算所有派生等离子体参数。"""
    w_pe = np.sqrt(n_e * q_e**2 / (m_e * epsilon_0))
    w_pi = np.sqrt(n_ion * q_e**2 / (ion_mass * epsilon_0))
    d_e = c / w_pe
    d_i = c / w_pi

    kT_J = T_e_eV * q_e
    theta_i = kT_J / (ion_mass * c**2)
    u_th_ion = np.sqrt(theta_i)   # v_th/c for non-relativistic ions

    return {
        "w_pe": w_pe, "w_pi": w_pi,
        "d_e": d_e,   "d_i": d_i,
        "kT_J": kT_J,
        "theta_i": theta_i,
        "u_th_ion": u_th_ion,
        "v_th_ion": u_th_ion * c,
    }


class SimulationParameters:
    """
    BBN 混合 PIC 模拟参数。

    物理参数全部由 CosmoCons.dat 数据驱动。
    用户只控制：温度选择、网格、步数、磁场。
    """

    # ---- 温度选择 ----
    cosmocons_row = 8  # CosmoCons.dat 行号 (0-based)

    # ---- 离子种类 ----
    ion_mass = m_p           # kg (质子)
    ion_charge_number = 1    # Z
    ion_mass_fraction = 1.0  # 质子占总重子质量的比例 (84keV 时接近 1)

    # ---- HybridPICSolver ----
    electron_gamma_eos = 1.0  # 等温 (净电子通过对背景快速热化)
    plasma_resistivity = 0.0  # η (理想 MHD)
    hybrid_substeps = 100

    # ---- 模拟维度 ----
    dim = 2

    # ---- 无量纲参数（以 d_i 和 1/ω_pi 为单位）----
    LX = 20.0
    LY = 20.0
    LZ = 20.0
    LT = 10.0   # 短测试
    DT = 0.05

    # ---- 数值参数 ----
    NX = 32
    NY = 32
    NZ = 32
    NPPC = 50

    # ---- 磁场 ----
    target_sigma = 0.0  # 0 = 无初始磁场

    # ---- 诊断 ----
    field_total_step = 5
    particle_total_step = 5
