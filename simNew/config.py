#!/usr/bin/env python3

class SimulationParameters:
    """
    这是一个专门用于存放所有用户可配置参数的类。
    通过修改这里的数值，您可以独立地控制模拟的各个方面。
    """
    # --- 1. 基础物理参数 (Independent Physical Parameters) ---
    # 这些参数现在描述一个相对论性的电子-正电子对等离子体

    # 设置一个与等效热能磁场 B_norm (根据先前模拟约 2.2e4 T) 可比拟的非零初始磁场
    B0 = 1.0e4  # 初始磁场强度 (T)

    n_plasma = 7.3e27  # 等离子体数密度 (m^-3) (这是指电子或正电子的数密度)
    T_plasma_eV = 8.4e4  # 等离子体温度 (eV), e.g., 1 MeV. 对电子和正电子相同。
    # 在80KeV这个温度，正负电子已经在大量湮灭了，
    n_photon_to_plasma_ratio = 3.93  # 光子与电子(或正电子)的数密度之比

    # --- 2. 无量纲模拟参数 (Dimensionless Simulation Setup) ---
    LX = 20.0  # 模拟域 x 方向长度 (单位: 电子趋肤深度 d_e)
    LY = 20.0  # 模拟域 y 方向长度 (单位: 电子趋肤深度 d_e)
    LZ = 20.0  # 模拟域 z 方向长度 (单位: 电子趋肤深度 d_e)
    LT = 400.0  # 模拟总时长 (单位: 等离子体周期 1/w_pe)
    DT = 0.05  # 时间步长 (单位: 等离子体周期 1/w_pe) (需满足CFL条件)

    # --- 3. 数值和扰动参数 (Numerical and Perturbation Parameters) ---
    NX = 32  # x 方向网格数
    NY = 32  # y 方向网格数
    NZ = 32  # z 方向网格数
    NPPC = 10  # 每个单元的宏粒子数 (每个物种)

    # 磁场和导向场设置
    Bg_ratio = 0  # 导向场与B0的比值
    dB_ratio = 0  # 初始扰动磁场与B0的比值

    # --- 4. 非热扰动参数 (Non-thermal Perturbation Parameters) --- #
    beam_fraction = 0.5  # 非热束流粒子占总数的比例 (e.g., 20%)
    beam_energy_eV = 8.4e4  # 束流粒子的动能 (eV)。例如 1.0e6 表示 1 MeV。

    # --- 5. 输出和诊断 ---
    output_dir = "测试"  # 默认输出目录