#!/usr/bin/env python3

class SimulationParameters:
    """
    BBN 混合方法模拟参数（占位）。
    后续将包含：离子种类、CosmoCons 数据路径、混合方法选项等。
    """
    # --- 运行控制 ---
    run_id = 0

    # --- 维度 ---
    dim = 2

    # --- 物理参数 ---
    n_plasma = 7.28e33  # 电子数密度 m^-3（T ≈ 84 keV, CosmoCons row 8）
    T_plasma_eV = 8.4e4  # 等离子体温度 eV

    # --- 离子参数 ---
    ion_mass_ratio = 1836  # m_p / m_e（可降低以加速模拟）
    ion_density_ratio = 1e-2  # n_p / n_e（人为放大，物理值 ~1e-9）

    # --- 无量纲模拟参数 ---
    LX = 20.0
    LY = 20.0
    LZ = 20.0
    LT = 100.0
    DT = 0.025

    # --- 数值参数 ---
    NX = 32
    NY = 32
    NZ = 32
    NPPC = 10

    # --- 功能开关 ---
    enable_collision = True
    enable_qed = False
    field_total_step = 40
    particle_total_step = 40
    ndt = 5
