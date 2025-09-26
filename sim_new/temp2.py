#!/usr/bin/env python3
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0

# =============================================================================
# 1. 从您的WarpX脚本中提取的参数
# =============================================================================

# --- 基础物理参数 ---
n_plasma = 7.3e27  # 等离子体数密度 (m^-3)
T_plasma_eV = 8.4e4  # 等离子体温度 (eV)

# --- 无量纲模拟参数 ---
LX = 40.0  # 模拟域 x 方向长度 (单位: d_e)
LZ = 20.0  # 模拟域 z 方向长度 (单位: d_e)
DT = 0.05  # 您选择的时间步长 (单位: 1/w_pe)

# --- 数值参数 ---
NX = 128  # x 方向网格数
NZ = 128  # z 方向网格数

# =============================================================================
# 2. 派生参数计算
# =============================================================================

# 等离子体频率 (rad/s)
w_pe = np.sqrt(n_plasma * e ** 2 / (m_e * epsilon_0))
# 电子趋肤深度 (m)
d_e = c / w_pe

# 物理域尺寸 (m)
Lx = LX * d_e
Lz = LZ * d_e

# 网格尺寸 (m)
dx = Lx / NX
dz = Lz / NZ

# 您当前选择的时间步长 (s)
dt_chosen = DT / w_pe


# =============================================================================
# 3. CFL 条件计算
# =============================================================================

def calculate_cfl_limit(dx, dz):
    """
    根据给定的网格尺寸计算最大允许的时间步长 (dt_max)。
    """
    # 从 CFL 公式反解 dt_max
    # dt_max < 1 / (c * sqrt(1/dx² + 1/dz²))
    dt_max = 1.0 / (c * np.sqrt(1 / dx ** 2 + 1 / dz ** 2))
    return dt_max


# 计算最大允许的 dt 和 DT
dt_max_stable = calculate_cfl_limit(dx, dz)
DT_max_stable = dt_max_stable * w_pe

# =============================================================================
# 4. 结果输出
# =============================================================================

if __name__ == "__main__":
    print("--- CFL 条件与时间步长检查器 ---")
    print(f"输入参数:")
    print(f"  网格: {NX} x {NZ}")
    print(f"  物理域: {Lx:.3e} m x {Lz:.3e} m")
    print(f"  网格尺寸: dx = {dx:.3e} m, dz = {dz:.3e} m")
    print("-" * 35)
    print("计算结果:")
    print(f"  最大稳定时间步长 (物理单位): dt_max = {dt_max_stable:.3e} s")
    print(f"  最大稳定时间步长 (归一化单位): DT_max = {DT_max_stable:.4f} / w_pe")
    print("-" * 35)
    print("您的设置:")
    print(f"  您选择的时间步长 (物理单位): dt     = {dt_chosen:.3e} s")
    print(f"  您选择的时间步长 (归一化单位): DT     = {DT:.4f} / w_pe")
    print("-" * 35)

    # 留出一点安全余量，通常求解器参数 warpx.cfl 会设为 0.99 左右
    safety_factor = 0.99
    if dt_chosen < dt_max_stable * safety_factor:
        print(f"✅ 结论: 您的时间步长 DT = {DT:.4f} 是安全的。")
        print(f"   它小于 {safety_factor * 100}% 的CFL极限 ({DT_max_stable:.4f})。")
    else:
        print(f"❌ 警告: 您的时间步长 DT = {DT:.4f} 可能会导致数值不稳定！")
        print(f"   它超过了 {safety_factor * 100}% 的CFL极限 ({DT_max_stable:.4f})。建议减小 DT。")