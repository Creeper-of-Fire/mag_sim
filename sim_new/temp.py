#!/usr/bin/env python3
import numpy as np
from scipy.constants import e, m_e, c, epsilon_0, k as k_B

# =============================================================================
# 1. 从您的WarpX脚本中提取的物理参数
# =============================================================================
# 等离子体数密度 (m^-3) (电子或正电子的密度)
n_plasma = 7.3e27
# 等离子体温度 (eV)
T_plasma_eV = 8.4e4


# =============================================================================
# 2. 物理计算函数
# =============================================================================

def calculate_coulomb_log(n_e, T_eV):
    """
    计算库仑对数 (Coulomb Logarithm, lnΛ)。
    这是描述一次碰撞中远距离小角度散射与近距离大角度散射相对重要性的参数。

    Args:
        n_e (float): 电子数密度 (m^-3)。
        T_eV (float): 电子温度 (eV)。

    Returns:
        float: 库仑对数值。
    """
    # 将温度从eV转换为开尔文
    T_K = T_eV * e / k_B

    # 计算德拜长度 (Debye Length)
    # 德拜长度是等离子体中电荷屏蔽的特征尺度。
    # 注意：对于对生等离子体，总电荷密度是 n_e + n_p = 2*n_e
    lambda_D = np.sqrt(epsilon_0 * k_B * T_K / (2 * n_e * e ** 2))

    # 计算90度散射的平均碰撞参数 (b_perp)
    # 这是两个粒子相互作用时，能产生90度偏转的最近距离。
    # 对于热分布，平均动能 <E_k> = 3/2 * kT
    b_perp = e ** 2 / (4 * np.pi * epsilon_0 * (3 * k_B * T_K))

    # 计算 Λ (Lambda)
    Lambda = lambda_D / b_perp

    print(lambda_D)

    if Lambda <= 1:
        print("警告：Λ <= 1，等离子体不是弱耦合的，库仑对数计算可能不准确。")
        return 0

    return np.log(Lambda)


def calculate_collision_time(n_e, T_eV, ln_Lambda):
    """
    计算电子-电子 (e-e) 碰撞的特征时间 (斯皮策公式)。
    这个时间代表一个“测试”电子因与其他“场”电子的多次小角度散射
    累积效应而使其速度方向偏转约90度所需的平均时间。

    Args:
        n_e (float): 电子数密度 (m^-3)。
        T_eV (float): 电子温度 (eV)。
        ln_Lambda (float): 库仑对数。

    Returns:
        float: 碰撞时间 (秒)。
    """
    # 将温度从eV转换为焦耳
    T_J = T_eV * e

    # 使用国际单位制下的斯皮策公式计算电子-电子碰撞频率 nu_ee
    # nu_ee = (n_e * e**4 * ln_Lambda) / (4 * np.pi * epsilon_0**2 * m_e**2 * v_th**3)
    # 其中热速度 v_th = sqrt(3 * kT / m_e)

    # 更直接的公式：
    # 因子 4*sqrt(2*pi) vs 4*sqrt(pi) 的区别在于对速度分布的平均方式不同，结果相似
    numerator = 4 * np.sqrt(2 * np.pi) * epsilon_0 ** 2 * np.sqrt(m_e) * (T_J) ** (3 / 2)
    denominator = n_e * e ** 4 * ln_Lambda

    tau_ee = numerator / denominator

    return tau_ee


# =============================================================================
# 3. 主程序
# =============================================================================

if __name__ == "__main__":
    print("--- 库仑碰撞时间计算器 ---")
    print(f"输入参数:")
    print(f"  等离子体密度 n_e = {n_plasma:.2e} m^-3")
    print(f"  等离子体温度 T_e = {T_plasma_eV:.2e} eV ({T_plasma_eV / 1e3:.1f} keV)")
    print("-" * 30)

    # 计算库仑对数
    ln_Lambda = calculate_coulomb_log(n_plasma, T_plasma_eV)
    print(f"计算结果:")
    print(f"  库仑对数 ln(Λ) = {ln_Lambda:.2f}")

    # 计算非相对论碰撞时间
    tau_collision_non_rel = calculate_collision_time(n_plasma, T_plasma_eV, ln_Lambda)
    print(f"  经典(非相对论)碰撞时间 τ_ee = {tau_collision_non_rel:.3e} s")

    print("\n--- 物理意义和相对论修正讨论 ---")

    # 1. 相对论参数 Theta
    theta = (T_plasma_eV * e) / (m_e * c ** 2)
    print(f"1. 相对论参数 Θ = kT / (m_e*c^2) = {theta:.3f}")
    if theta > 0.1:
        print("   分析: Θ > 0.1，系统处于“轻度相对论”状态。电子的热动能已不可忽略其静止质量能。")
        # 估算相对论修正
        # 相对论效应使得粒子更“硬”，难以偏转，碰撞截面减小，因此碰撞时间会变长。
        # 一个粗略的估算是将碰撞时间乘以 gamma^(3/2) 或 gamma^2。
        # 对于热分布，平均gamma因子约可估算为 1 + 15/8 * theta (对于gamma=5/3)
        # 这里我们用一个更简单的估算 gamma ~ 1 + theta
        gamma_est = 1 + theta
        tau_collision_rel_est = tau_collision_non_rel * (gamma_est ** 1.5)
        print(f"   因此，经典公式会低估真实的碰撞时间。")
        print(f"   基于平均gamma ~ {gamma_est:.2f} 的一个粗略估计：")
        print(f"   相对论修正后的碰撞时间 τ_ee' ≈ {tau_collision_rel_est:.3e} s")
    else:
        print("   分析: Θ <= 0.1，非相对论近似是比较合理的。")

    # 2. 与等离子体周期的比较
    w_pe = np.sqrt(n_plasma * e ** 2 / (m_e * epsilon_0))
    plasma_period = 1 / w_pe
    print(f"\n2. 与模拟基本时间尺度的比较:")
    print(f"   等离子体频率 w_pe = {w_pe:.3e} rad/s")
    print(f"   等离子体周期 1/w_pe = {plasma_period:.3e} s")

    # 您的模拟时间步长 DT=0.05, 总时长 LT=400.0 (以 1/w_pe 为单位)
    sim_dt = 0.05 * plasma_period
    sim_total_time = 400.0 * plasma_period

    collision_freq_param = w_pe * tau_collision_non_rel  # 无量纲碰撞频率

    print(f"   您的模拟总时长 T_sim = {sim_total_time:.3e} s")
    print(f"\n   关键无量纲参数 w_pe * τ_ee ≈ {collision_freq_param:.2f}")

    print("\n--- 结论 ---")
    if collision_freq_param > 1000:
        print(f"碰撞时间 ({tau_collision_non_rel:.1e} s) 远大于模拟总时长 ({sim_total_time:.1e} s)。")
        print("结论: 在这个模拟的时间尺度内，该等离子体系统是高度“无碰撞的”(collisionless)。")
        print("粒子动力学将由电磁场主导，而不是二进制碰撞。这符合磁重联PIC模拟的基本假设。")
    elif collision_freq_param > 10:
        print(f"碰撞时间 ({tau_collision_non_rel:.1e} s) 大于模拟总时长 ({sim_total_time:.1e} s)。")
        print("结论: 系统是“弱碰撞的”(weakly collisional)。碰撞效应可能在模拟后期或对特定能量分布的粒子产生一些影响。")
    else:
        print(f"碰撞时间 ({tau_collision_non_rel:.1e} s) 与模拟总时长 ({sim_total_time:.1e} s) 相当或更短。")
        print("结论: 系统是“碰撞主导的”(collisional)。二进制碰撞在能量和动量交换中扮演重要角色，会显著影响磁重联的演化。")