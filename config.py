#!/usr/bin/env python3

class SimulationParameters:
    """
    这是一个专门用于存放所有用户可配置参数的类。0
.
    12311111111113131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131311通过修改这里的数值，您可以独立地控制模拟的各个方面。
    """
    # --- 1. 基础物理参数 (Independent Physical Parameters) ---
    # 这些参数现在描述一个相对论性的电子-正电子对等离子体

    # 目标磁化率 (Sigma)
    # 定义: 磁能密度 / 粒子热焓密度 (Magnetic Energy / Enthalpy)
    # 作用: 如果此值 > 0，程序将自动根据等离子体参数计算所需的 B_rms，并忽略下方的 B0 设置。
    #      如果设置为 -1.0，则回退到使用下方的 B0 作为固定磁场强度。
    # 物理含义速查:
    #   0.1 : 磁能占总能量的 ~9%。热压力主导 (High Beta)。
    #   1.0 : 磁能占总能量的 50%。能量均分。
    #   10.0: 磁能占总能量的 ~91%。磁能主导 (Low Beta)，利于相对论性加速。
    #   -1.0: (不使用能量比) 强制使用下方指定的 B0 绝对值。
    target_sigma = 0.1

    # 设置一个与等效热能磁场 B_norm (根据先前模拟约 2.2e4 T) 可比拟的非零初始磁场
    # 作用: 仅当 target_sigma <= 0 时生效。
    #      如果 B_field_type 是 Gaussian，这将被视为目标 RMS (均方根) 强度。
    B0 = 1.0e4

    n_plasma = 7.3e33  # 等离子体数密度 (m^-3) (这是指电子或正电子的数密度)
    T_plasma_eV = 8.4e4  # 等离子体温度 (eV), e.g., 1 MeV. 对电子和正电子相同。
    # 在80KeV这个温度，正负电子已经在大量湮灭了，
    n_photon_to_plasma_ratio = 3.93  # 光子与电子(或正电子)的数密度之比

    # --- 2. 无量纲模拟参数 (Dimensionless Simulation Setup) ---
    LX = 20.0  # 模拟域 x 方向长度 (单位: 电子趋肤深度 d_e)
    LY = 20.0  # 模拟域 y 方向长度 (单位: 电子趋肤深度 d_e)
    LZ = 20.0  # 模拟域 z 方向长度 (单位: 电子趋肤深度 d_e)
    LT = 100.0  # 模拟总时长 (单位: 等离子体周期 1/w_pe)
    DT = 0.025  # 时间步长 (单位: 等离子体周期 1/w_pe) (需满足CFL条件)

    # --- 3. 数值和扰动参数 (Numerical and Perturbation Parameters) ---
    NX = 32  # x 方向网格数
    NY = 32  # y 方向网格数
    NZ = 32  # z 方向网格数
    NPPC = 10  # 每个单元的宏粒子数 (每个物种)

    # 磁场和导向场设置
    Bg_ratio = 0.  # 导向场与B0的比值
    dB_ratio = 0.  # 初始扰动磁场与B0的比值

    # --- 4. 非热扰动参数 (Non-thermal Perturbation Parameters) --- #
    beam_fraction = 0.  # 非热束流粒子占总数的比例 (e.g., 20%)
    beam_energy_eV = 8.4e4  # 束流粒子的动能 (eV)。例如 1.0e6 表示 1 MeV。

    # --- 5. 输出和诊断 ---
    output_dir = "sim_result/测试4"  # 默认输出目录

    # --- 6. 磁场配置 ---
    # 在这里，您可以选择背景磁场的类型
    # 'uniform': 均匀磁场 (原始行为)
    # 'single_gaussian': 单个随机位置、随机方向的高斯磁场
    # 'multi_gaussian': 多个随机高斯磁场的叠加
    B_field_type = "multi_gaussian"

    # 当 B_field_type 为 'multi_gaussian' 时，此参数指定高斯场的数量
    num_gaussians = 5

    # 高斯包的宽度，以相应方向的模拟域尺寸 (Lx, Ly, Lz) 的分数表示。
    # 例如，0.1 表示宽度为 0.1 * Lx。较小的值产生更集中的磁场结构。
    gaussian_width_L_ratio = 0.15