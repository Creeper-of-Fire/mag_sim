#!/usr/bin/env python3
"""
PIC 模拟约束下的 sigma 参数空间探索。

核心问题：当 target_sigma 增大 → T_new 下降，B 上升，
模拟面临三个关键约束：
  1. Debye 长度分辨率   (λ_D / Δx)
  2. 热回旋半径分辨率   (r_g / Δx)
  3. 碰撞频率子步进限制 (ν_coll * DT * ndt)
同时要关注物理时间尺度的变化（等离子体变得多"碰撞性"）。
"""

import math
import numpy as np
from scipy.constants import m_e, c, e, mu_0, epsilon_0, k as kb
from scipy.special import kve as bessel_kve

# ============================================================
# 物理常数
# ============================================================
MC2_J = m_e * c ** 2
MC2_eV = MC2_J / e

# ============================================================
# 恒能量分配计算（同 csv_tool_constant_energy.py）
# ============================================================

def calc_constant_energy(T_ref_eV: float, sigma: float, n_plasma: float = 7.28e33):
    """返回 (T_new_eV, B_new_T) 或 (None, None) 不可行"""
    if sigma < 0:
        return None, None
    kb_T_ref_J = T_ref_eV * e
    term1 = (MC2_J + 3.0 * kb_T_ref_J) / (1.0 + sigma)
    kb_T_new_J = (term1 - MC2_J) / 3.0
    if kb_T_new_J < 0:
        return None, None
    T_new_eV = kb_T_new_J / e
    U_p_new = 2 * n_plasma * (MC2_J + 3.0 * kb_T_new_J)
    U_B_new = sigma * U_p_new
    B_new = math.sqrt(2 * mu_0 * U_B_new)
    return T_new_eV, B_new


# ============================================================
# 等离子体基本参数
# ============================================================

def plasma_params(n: float, T_eV: float, B_T: float):
    """根据 n, T, B 计算无量纲等离子体参数"""
    omega_pe = math.sqrt(n * e**2 / (m_e * epsilon_0))      # rad/s
    omega_ce = e * B_T / m_e if B_T > 0 else 1e-30          # rad/s (避免除零)
    d_e = c / omega_pe                                      # skin depth (m)

    # Debye 长度 (对于相对论等离子体依然成立，因为 λ_D² = ε₀ T / (n e²))
    T_J = T_eV * e   # T in Joules
    lambda_D = math.sqrt(epsilon_0 * T_J / (n * e**2))  # m

    # 非相对论热速度近似 (用于估计)
    v_th = min(c, math.sqrt(3 * T_J / m_e))   # capped at c

    # 热回旋半径 (非相对论近似)
    r_g = v_th / omega_ce if omega_ce > 1 else 1e30

    # 相对论参数 z = m_e c² / T
    z = MC2_eV / T_eV if T_eV > 0 else 1e10

    # 等离子体参数 g = 1/(n λ_D³)
    g = 1.0 / (n * lambda_D**3) if lambda_D > 0 else 0

    # ω_pe * dt (collision step check)
    coll_step_norm = 1.0  # placeholder

    return {
        "omega_pe": omega_pe,              # rad/s
        "omega_ce": omega_ce,              # rad/s
        "d_e_m": d_e,                      # m
        "lambda_D_m": lambda_D,            # m
        "v_th_m_s": v_th,                  # m/s
        "r_g_m": r_g,                      # m
        "z": z,                            # mc²/T, >1 non-rel, <1 rel
        "plasma_parameter": g,             # g << 1 means collective
    }


# ============================================================
# 相对论碰撞频率 (Braams & Karney 1987, 同 coll.py)
# ============================================================

def thermal_collision_freq(n: float, T_eV: float, coulomb_log: float = 15.0):
    """
    e-e 热弛豫速率 ν_rel (s⁻¹)。
    计算背景热电子之间的碰撞频率，使用 Braams & Karney 完整相对论公式。
    """
    T_J = T_eV * e
    z = MC2_J / T_J  # z = mc²/T

    if z > 200:  # 非相对论极限，用经典公式
        v_th = math.sqrt(3 * T_J / m_e)
        nu_ee = (n * e**4 * coulomb_log) / (4 * math.pi * epsilon_0**2 * m_e**2 * v_th**3)
        return nu_ee

    # 相对论修正 (Braams & Karney)
    Gamma_ee = (n * e**4 * coulomb_log) / (4 * math.pi * epsilon_0**2 * m_e**2)
    u_tb_sq = T_J / m_e

    K0 = bessel_kve(0, z)
    K1 = bessel_kve(1, z)
    K2 = bessel_kve(2, z)

    # 热平均碰撞频率近似 (从 test-particle 公式推导)
    # 使用 3T 作为热粒子的特征能量
    E_test_J = 3 * T_J  # 典型热能粒子
    gamma = 1.0 + E_test_J / MC2_J
    if gamma <= 1.0:
        return 1e30
    v = c * math.sqrt(1.0 - 1.0 / gamma**2)
    if v < 1e-10:
        return 1e30

    # D_uu 在热粒子处
    term_bessel_1 = K1 / K2
    term_bessel_2 = K0 / K1
    bracket = 1.0 - term_bessel_2 * (u_tb_sq / (gamma**2 * c**2))
    D_uu = Gamma_ee * term_bessel_1 * (u_tb_sq / v**3) * bracket
    if D_uu <= 0:
        return 1e30

    F_u = -(m_e * v / T_J) * D_uu
    u = gamma * v
    tau = u / abs(F_u) if abs(F_u) > 1e-50 else 1e30
    return 1.0 / tau


def gyro_average_freq(omega_ce: float):
    """回旋频率 (Hz)"""
    return omega_ce / (2 * math.pi)


# ============================================================
# 模拟参数设置 (和已有 run 一致)
# ============================================================

SIM_DEFAULTS = {
    "LX": 100.0, "LY": 100.0, "LZ": 100.0,
    "NX": 256, "NY": 256, "NZ": 256,
    "DT": 0.2, "LT": 1000.0,
    "dim": 2,
    "ndt": 5,
}


# ============================================================
# 主约束分析
# ============================================================

def analyze_constraints(T_ref_eV: float, sigma: float, sim: dict, n_plasma: float = 7.28e33, coulomb_log: float = 15.0):
    """对单组 (T_ref, sigma) 分析所有模拟约束"""
    T_new, B_new = calc_constant_energy(T_ref_eV, sigma, n_plasma)
    if T_new is None:
        return None

    dx = sim["LX"] / sim["NX"]  # normalized units (d_e)

    # 等离子体参数
    pp = plasma_params(n_plasma, T_new, B_new)
    nu_coll = thermal_collision_freq(n_plasma, T_new, coulomb_log)

    # ---- 约束 1: Debye 长度分辨率 ----
    # λ_D / d_e = sqrt(T_new / 511000)
    lambda_D_over_de = math.sqrt(T_new / 511000.0) if T_new < 511000 else 1.0
    lambda_D_over_dx = lambda_D_over_de / dx

    # ---- 约束 2: 热回旋半径分辨率 ----
    # r_g/d_e = v_th/c / (ω_ce/ω_pe)
    omega_pe = pp["omega_pe"]
    omega_ce = pp["omega_ce"]
    r_g_over_de = (pp["v_th_m_s"] / c) / (omega_ce / omega_pe) if omega_ce > 0 else 1e30
    r_g_over_dx = r_g_over_de / dx

    # ---- 约束 3: 碰撞频率 ----
    nu_coll_over_omega_pe = nu_coll / omega_pe
    coll_per_dt = nu_coll_over_omega_pe * sim["DT"]  # 每个时间步的碰撞概率
    coll_per_ndt_step = coll_per_dt * sim["ndt"]      # 每个碰撞子步的碰撞概率

    # ---- 物理时间尺度 ----
    tau_coll_s = 1.0 / nu_coll if nu_coll > 0 else 1e30
    tau_sim_s = sim["LT"] / omega_pe                    # 模拟总时长 (s)
    tau_gyro_s = 2 * math.pi / omega_ce if omega_ce > 0 else 1e30
    n_gyro_per_sim = tau_sim_s / tau_gyro_s if tau_gyro_s > 0 else 0

    coll_ratio = tau_coll_s / tau_sim_s
    # coll_ratio > 1 => 碰撞时间比模拟长 => 碰撞不显著
    # coll_ratio < 1 => 模拟期间发生多次碰撞 => 碰撞性

    return {
        "sigma": sigma,
        "T_new_eV": T_new,
        "B_T": B_new,
        # 约束 1: Debye
        "lambda_D_over_dx": lambda_D_over_dx,
        # 约束 2: 回旋
        "r_g_over_dx": r_g_over_dx,
        "r_g_m": pp["r_g_m"],
        # 约束 3: 碰撞
        "nu_coll_Hz": nu_coll,
        "nu_coll_over_omega_pe": nu_coll_over_omega_pe,
        "coll_per_ndt_step": coll_per_ndt_step,
        # 物理时间尺度
        "tau_coll_vs_sim": coll_ratio,
        "tau_coll_s": tau_coll_s,
        "tau_sim_s": tau_sim_s,
        "tau_gyro_s": tau_gyro_s,
        "n_gyro_per_sim": n_gyro_per_sim,
        # 相对论参数
        "z": pp["z"],
        "plasma_param": pp["plasma_parameter"],
        # 无量纲化
        "lambda_D_over_de": lambda_D_over_de,
        "r_g_over_de": r_g_over_de,
        "T_keV": T_new / 1000,
    }


# ============================================================
# 约束判据
# ============================================================

def evaluate_feasibility(c: dict):
    """
    根据经验判据判断该参数组合是否可行。
    返回: (is_feasible: bool, 最紧约束: str, 警告列表: list)
    """
    warnings = []

    # 1. Debye 长度 (必须 > 0.3, 推荐 > 0.5)
    if c["lambda_D_over_dx"] < 0.2:
        warnings.append(f"❌ λ_D/Δx = {c['lambda_D_over_dx']:.3f} < 0.2 → Debye 严重未分辨，数值加热不可接受")
    elif c["lambda_D_over_dx"] < 0.35:
        warnings.append(f"⚠️  λ_D/Δx = {c['lambda_D_over_dx']:.3f} < 0.35 → Debye 欠分辨，可能有数值加热")
    else:
        warnings.append(f"✅ λ_D/Δx = {c['lambda_D_over_dx']:.3f} ≥ 0.35 → Debye 可分辨")

    # 2. 回旋半径 (至少 > 0.1，最好 > 0.3)
    if c["r_g_over_dx"] < 0.1:
        warnings.append(f"❌ r_g/Δx = {c['r_g_over_dx']:.4f} < 0.1 → 回旋运动完全未分辨")
    elif c["r_g_over_dx"] < 0.3:
        warnings.append(f"⚠️  r_g/Δx = {c['r_g_over_dx']:.4f} < 0.3 → 回旋运动部分分辨")
    else:
        warnings.append(f"✅ r_g/Δx = {c['r_g_over_dx']:.4f} ≥ 0.3 → 回旋运动可分辨")

    # 3. 碰撞子步进 (coll_per_ndt_step < 1)
    if c["coll_per_ndt_step"] > 1.0:
        warnings.append(f"❌ coll_step = {c['coll_per_ndt_step']:.2e} >> 1 → 碰撞子步进严重不足 (需增大 ndt)")
    elif c["coll_per_ndt_step"] > 0.1:
        warnings.append(f"⚠️  coll_step = {c['coll_per_ndt_step']:.4f} > 0.1 → 碰撞频率偏高，建议增大 ndt")
    else:
        warnings.append(f"✅ coll_step = {c['coll_per_ndt_step']:.4f} ≤ 0.1 → 碰撞子步进足够")

    # 4. 碰撞性 (tau_coll vs tau_sim)
    if c["tau_coll_vs_sim"] < 0.1:
        warnings.append(f"⚡ 碰撞性: τ_coll/τ_sim = {c['tau_coll_vs_sim']:.3e} << 1 → 高度碰撞性等离子体")
    elif c["tau_coll_vs_sim"] < 1.0:
        warnings.append(f"⚡ 碰撞性: τ_coll/τ_sim = {c['tau_coll_vs_sim']:.3f} < 1 → 中等碰撞性")
    elif c["tau_coll_vs_sim"] < 10:
        warnings.append(f"⚡ 碰撞性: τ_coll/τ_sim = {c['tau_coll_vs_sim']:.2f} → 弱碰撞性")
    else:
        warnings.append(f"⚡ 碰撞性: τ_coll/τ_sim = {c['tau_coll_vs_sim']:.2e} >> 1 → 无碰撞")

    # 5. 每模拟周期回旋数
    if c["n_gyro_per_sim"] < 10:
        warnings.append(f"💫 n_gyro = {c['n_gyro_per_sim']:.1f} → 回旋数太少")

    # 判断整体可行性
    critical = [w for w in warnings if w.startswith("❌")]
    is_feasible = len(critical) == 0
    tightest = critical[0] if critical else (warnings[0] if warnings else "OK")

    return is_feasible, tightest, warnings


# ============================================================
# 表格打印
# ============================================================

def print_constraint_table(T_ref_eV: float, sigmas: list, sim: dict, n_plasma: float = 7.28e33):
    print(f"\n{'='*110}")
    print(f"  T_ref = {T_ref_eV:.1f} eV ({T_ref_eV/1000:.2f} keV)  |  "
          f"Δx = {sim['LX']/sim['NX']:.4f} d_e  |  "
          f"DT = {sim['DT']}  |  ndt = {sim['ndt']}")
    print(f"  Grid: {sim['NX']}³, LX={sim['LX']} d_e  |  LT={sim['LT']}  |  n_plasma={n_plasma:.2e}")
    print(f"{'='*110}")

    print(f"{'sigma':>10} | {'T_new(keV)':>10} | {'B(T)':>11} | {'λ_D/Δx':>8} | {'r_g/Δx':>8} | "
          f"{'ν_coll/ω_pe':>11} | {'coll_step':>9} | {'τ_coll/τ_sim':>12} | {'特征':>16}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*11}-+-{'-'*9}-+-{'-'*12}-+-{'-'*16}")

    assessments = []
    for sigma in sigmas:
        c = analyze_constraints(T_ref_eV, sigma, sim, n_plasma)
        if c is None:
            print(f"{sigma:>10.6f} | {'—':>10s} | {'—':>11s} | {'—':>8s} | {'—':>8s} | "
                  f"{'—':>11s} | {'—':>9s} | {'—':>12s} | {'不可行':>16s}")
            continue

        feasible, tightest, warns = evaluate_feasibility(c)
        # 取精简特征描述
        if c["z"] > 10:
            regime = "冷等离子体"
        elif c["z"] > 1:
            regime = "非相对论"
        elif c["z"] > 0.3:
            regime = "弱相对论"
        else:
            regime = "相对论性"

        coll_note = ""
        if c["tau_coll_vs_sim"] < 0.1:
            coll_note = "强碰撞"
        elif c["tau_coll_vs_sim"] < 1:
            coll_note = "有碰撞"
        elif c["tau_coll_vs_sim"] < 10:
            coll_note = "弱碰撞"
        else:
            coll_note = "无碰撞"

        print(f"{sigma:>10.6f} | {c['T_keV']:>10.4f} | {c['B_T']:>11.4e} | "
              f"{c['lambda_D_over_dx']:>8.4f} | {c['r_g_over_dx']:>8.4f} | "
              f"{c['nu_coll_over_omega_pe']:>11.4e} | {c['coll_per_ndt_step']:>9.4f} | "
              f"{c['tau_coll_vs_sim']:>12.4e} | {regime+','+coll_note:>16s}")

        assessments.append((sigma, c['T_keV'], c['B_T'], feasible, tightest, warns))

    return assessments


# ============================================================
# sigma 列表
# ============================================================

SIGMAS_FULL = [
    0.0001, 0.0005, 0.001, 0.005, 0.01,
    0.02, 0.03, 0.04, 0.05,
    0.06, 0.08, 0.10,
    0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.49,
]

SIGMAS_FOCUSED = [
    0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
]


# ============================================================
# 推荐组合并给出最紧约束
# ============================================================

def recommend_optimal(T_ref_eV: float, sim: dict, n_plasma: float = 7.28e33):
    """
    扫描 sigma，找出在当前模拟设置下可行的推荐范围。
    """
    print(f"\n{'='*80}")
    print(f"  推荐分析: T_ref = {T_ref_eV/1000:.1f} keV")
    print(f"{'='*80}")

    results = []
    for sigma in SIGMAS_FULL:
        c = analyze_constraints(T_ref_eV, sigma, sim, n_plasma)
        if c is None:
            continue
        feasible, tightest, warns = evaluate_feasibility(c)
        results.append((sigma, c["T_keV"], c["B_T"], feasible, tightest))

    feasible_ones = [r for r in results if r[3]]
    if feasible_ones:
        print(f"  可行的 sigma 范围: {feasible_ones[0][0]:.6f} ~ {feasible_ones[-1][0]:.6f}")
        print(f"  对应 T_new: {feasible_ones[0][1]:.2f} keV ~ {feasible_ones[-1][1]:.2f} keV")
        print(f"  对应 B: {feasible_ones[0][2]:.2e} T ~ {feasible_ones[-1][2]:.2e} T")
        print(f"\n  推荐参数点:")
        print(f"  {'sigma':>8} | {'T(keV)':>8} | {'B(T)':>12} | {'最紧约束':>30}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*30}")
        for r in feasible_ones:
            tag = r[4][:30] if len(r[4]) <= 30 else r[4][:27]+"..."
            print(f"  {r[0]:>8.6f} | {r[1]:>8.4f} | {r[2]:>12.4e} | {tag:>30}")
    else:
        print("  ❌ 在当前的模拟设置下没有可行的 sigma 值！")


# ============================================================
# 多温度对比
# ============================================================

def compare_grids(T_ref_eV: float, sigmas: list, n_plasma: float = 7.28e33):
    """
    对比不同网格设置下的约束。
    """
    grids = [
        ("256³, Δx=0.391", {"NX": 256, "LX": 100, "DT": 0.2, "LT": 1000, "ndt": 5}),
        ("128³, Δx=0.391", {"NX": 128, "LX": 50,  "DT": 0.2, "LT": 1000, "ndt": 5}),
        ("512³, Δx=0.195", {"NX": 512, "LX": 100, "DT": 0.2, "LT": 1000, "ndt": 5}),
    ]

    print(f"\n{'='*110}")
    print(f"  网格对比: T_ref = {T_ref_eV/1000:.1f} keV")
    print(f"{'='*110}")

    for label, sim in grids:
        print(f"\n--- {label} ---")
        print(f"{'sigma':>8} | {'T(keV)':>8} | {'λ_D/Δx':>8} | {'r_g/Δx':>8} | {'coll_step':>9} | {'可行?':>6}")
        print(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}")
        for sigma in sigmas:
            c = analyze_constraints(T_ref_eV, sigma, sim, n_plasma)
            if c is None:
                print(f"{sigma:>8.6f} | {'—':>8s} | {'—':>8s} | {'—':>8s} | {'—':>9s} | {'—':>6s}")
                continue
            feasible, _, _ = evaluate_feasibility(c)
            flag = "✅" if feasible else "❌"
            print(f"{sigma:>8.6f} | {c['T_keV']:>8.4f} | {c['lambda_D_over_dx']:>8.4f} | "
                  f"{c['r_g_over_dx']:>8.4f} | {c['coll_per_ndt_step']:>9.4f} | {flag:>6}")


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PIC 约束下 sigma 空间探索")

    parser.add_argument("--temperatures", type=float, nargs="+", default=None,
                        help="T_ref 列表 (eV)")
    parser.add_argument("--sigmas", type=float, nargs="+", default=None,
                        help="sigma 列表")
    parser.add_argument("--n-plasma", type=float, default=7.28e33)
    parser.add_argument("--compare-grids", action="store_true",
                        help="对比不同网格分辨率的约束")
    parser.add_argument("--coulomb-log", type=float, default=15.0)

    args = parser.parse_args()

    temperatures = args.temperatures or [84480.0, 58000.0, 30000.0, 12000.0]
    sigmas = args.sigmas or SIGMAS_FOCUSED

    for T_ref in temperatures:
        print_constraint_table(T_ref, sigmas, SIM_DEFAULTS, args.n_plasma)
        recommend_optimal(T_ref, SIM_DEFAULTS, args.n_plasma)

    if args.compare_grids:
        for T_ref in temperatures[:2]:
            compare_grids(T_ref, sigmas[:10], args.n_plasma)


if __name__ == "__main__":
    main()
