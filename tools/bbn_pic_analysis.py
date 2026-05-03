#!/usr/bin/env python3
"""
BBN - PIC 模拟参数映射与约束分析。

流程：
1. 读取 CosmoCons.dat（BBN 各时期参数：T9, time, Ne 等）
2. 将 T9 → T_plasma_eV，Ne → n_plasma（按参考点归一化）
3. 对每个 epoch + sigma 组合，计算 PIC 约束：
   - Debye 长度分辨率 (λ_D/Δx)
   - 回旋半径分辨率 (r_g/Δx)
   - 碰撞频率 / 子步进 (ν_coll/ω_pe)
4. 推荐可行参数组合
"""

import math
import sys
from pathlib import Path
from scipy.constants import m_e, c, e, mu_0, epsilon_0, k as kb
from scipy.special import kve as bessel_kve

# ============================================================
# 物理常数
# ============================================================
MC2_J = m_e * c ** 2
MC2_eV = MC2_J / e
K_eV_K = kb / e                    # Boltzmann → eV/K: 8.617e-5 eV/K

# ============================================================
# 1. 读取 CosmoCons.dat
# ============================================================

COSMO_PATH = Path("/mnt/d/User/Desktop/Project/BBNHHe/CosmoCons.dat")

def read_cosmocons(path: str = None):
    """
    返回 BBN 时期列表，每个时期是一个 dict:
      t9       - 温度 (10^9 K)
      time_s   - 时间 (s)
      Ne_cm3   - 电子数密度 (cm^-3)
      T_eV     - 温度 (eV)  = t9 * 10^9 * K_eV_K
    """
    path = path or str(COSMO_PATH)
    epochs = []
    with open(path) as f:
        for line in f:
            if line.strip().startswith("-") or line.strip().startswith("Temp"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            t9 = float(parts[0])
            time_s = float(parts[1])
            ne_cm3 = float(parts[9])
            t_eV = t9 * 1e9 * K_eV_K
            epochs.append({
                "t9": t9,
                "time_s": time_s,
                "Ne_cm3": ne_cm3,
                "T_eV": t_eV,
            })
    return epochs


# ============================================================
# 2. 参考点定义（当前模拟的默认参数）
# ============================================================

# 当前模拟在 ~84.48 keV 下使用 n_plasma = 7.28e33 m^-3
# CosmoCons.dat 中对应 T9 ≈ 0.980 (内插值)
REF_T_eV = 84480.0
REF_n_m3 = 7.28e33

# CosmoCons 中内插 Ne 在 T=84.48 keV → T9=0.980
# 利用 T9=1.687 (Ne=7.150e32) 和 T9=0.734 (Ne=5.718e31) 之间 T^3 内插
def interpolate_ne(t9_target, epochs):
    """基于 T³ 标度关系内插 Ne"""
    # 找 target 两边的点
    below = None
    above = None
    for ep in epochs:
        if ep["t9"] <= t9_target:
            below = ep
        if ep["t9"] >= t9_target and above is None:
            above = ep
    if below is None:
        return epochs[0]["Ne_cm3"] * (t9_target / epochs[0]["t9"])**3
    if above is None:
        return epochs[-1]["Ne_cm3"] * (t9_target / epochs[-1]["t9"])**3
    if below == above:
        return below["Ne_cm3"]
    # T³ 内插 (两边取权重)
    # Ne ∝ T³, 所以 Ne_target = Ne_below * (t9_target/t9_below)³
    # 用 closer 的点做内插
    ratio_b = (t9_target / below["t9"])**3
    ratio_a = (t9_target / above["t9"])**3
    ne_from_b = below["Ne_cm3"] * ratio_b
    ne_from_a = above["Ne_cm3"] * ratio_a
    return (ne_from_b + ne_from_a) / 2

# 计算参考点的 Ne 并确定归一化因子
EPOCHS = read_cosmocons()
REF_T9 = REF_T_eV / (1e9 * K_eV_K)
REF_Ne_cm3 = interpolate_ne(REF_T9, EPOCHS)
REF_Ne_m3 = REF_Ne_cm3 * 1e6  # cm⁻³ → m⁻³
NORM_FACTOR = REF_n_m3 / REF_Ne_m3

def n_plasma_from_epoch(epoch):
    """根据 BBN 时期的 Ne 计算模拟使用的 n_plasma (m⁻³)"""
    ne_m3 = epoch["Ne_cm3"] * 1e6
    return ne_m3 * NORM_FACTOR


# ============================================================
# 3. 恒能量分配计算
# ============================================================

def calc_constant_energy(T_ref_eV: float, sigma: float, n_plasma: float):
    """返回 (T_new_eV, B_new_T) 或 (None, None)"""
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
# 4. 碰撞频率 (Braams & Karney 相对论)
# ============================================================

def thermal_collision_freq(n_m3: float, T_eV: float, coulomb_log: float = 15.0):
    """相对论 e-e 热弛豫频率 ν_coll (s⁻¹)"""
    T_J = T_eV * e
    z = MC2_J / T_J

    if z > 200:  # 非相对论极限
        v_th = math.sqrt(3 * T_J / m_e)
        return (n_m3 * e**4 * coulomb_log) / (4 * math.pi * epsilon_0**2 * m_e**2 * v_th**3)

    Gamma_ee = (n_m3 * e**4 * coulomb_log) / (4 * math.pi * epsilon_0**2 * m_e**2)
    u_tb_sq = T_J / m_e

    K0 = bessel_kve(0, z)
    K1 = bessel_kve(1, z)
    K2 = bessel_kve(2, z)

    E_test_J = 3 * T_J
    gamma = 1.0 + E_test_J / MC2_J
    if gamma <= 1.0:
        return 1e30
    v = c * math.sqrt(1.0 - 1.0 / gamma**2)
    if v < 1e-10:
        return 1e30

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


# ============================================================
# 5. 约束分析
# ============================================================

SIM_DEFAULTS = {
    "LX": 100.0, "NX": 256,
    "DT": 0.2, "LT": 1000.0,
    "ndt": 5,
}

def analyze(epoch, sigma, sim=None):
    """
    对一个 BBN 时期 + sigma 组合做完整约束分析。
    返回 dict 或 None（不可行）。
    """
    sim = sim or SIM_DEFAULTS
    T_ref = epoch["T_eV"]
    n = n_plasma_from_epoch(epoch)
    dx = sim["LX"] / sim["NX"]

    T_new, B = calc_constant_energy(T_ref, sigma, n)
    if T_new is None:
        return None

    ω_pe = math.sqrt(n * e**2 / (m_e * epsilon_0))
    ω_ce = e * B / m_e if B > 1 else 1e-30
    ν_coll = thermal_collision_freq(n, T_new)

    # Debye 长度
    λ_D_over_de = math.sqrt(T_new / 511000.0) if T_new < 511000e3 else 1.0
    λ_D_over_dx = λ_D_over_de / dx

    # 热回旋半径
    v_th = min(c, math.sqrt(3 * T_new * e / m_e))
    r_g_over_de = (v_th / c) / (ω_ce / ω_pe) if ω_ce > 1e-10 else 1e30
    r_g_over_dx = r_g_over_de / dx

    # 碰撞
    ν_coll_over_ω_pe = ν_coll / ω_pe
    coll_step = ν_coll_over_ω_pe * sim["DT"] * sim["ndt"]

    # 时间尺度
    τ_coll = 1.0 / ν_coll if ν_coll > 0 else 1e30
    τ_sim = sim["LT"] / ω_pe
    τ_gyro = 2 * math.pi / ω_ce if ω_ce > 1e-10 else 1e30

    return {
        "T9": epoch["t9"],
        "time_s": epoch["time_s"],
        "T_ref_eV": T_ref,
        "T_new_eV": T_new,
        "B_T": B,
        "n_m3": n,
        "sigma": sigma,
        # 约束
        "λ_D_over_dx": λ_D_over_dx,
        "r_g_over_dx": r_g_over_dx,
        "ν_coll_over_ω_pe": ν_coll_over_ω_pe,
        "coll_step": coll_step,
        # 物理
        "τ_coll_vs_τ_sim": τ_coll / τ_sim,
        "ω_pe": ω_pe,
        "d_e_m": c / ω_pe,
    }


# ============================================================
# 6. 判据
# ============================================================

def evaluate(c):
    """返回 (feasible, warnings[])"""
    warns = []
    if c["λ_D_over_dx"] < 0.2:
        warns.append(f"❌ Debye λ_D/Δx={c['λ_D_over_dx']:.3f} < 0.2")
    elif c["λ_D_over_dx"] < 0.35:
        warns.append(f"⚠ Debye λ_D/Δx={c['λ_D_over_dx']:.3f} < 0.35")
    else:
        warns.append(f"✅ Debye λ_D/Δx={c['λ_D_over_dx']:.3f}")

    if c["r_g_over_dx"] < 0.1:
        warns.append(f"❌ 回旋 r_g/Δx={c['r_g_over_dx']:.4f} < 0.1")
    elif c["r_g_over_dx"] < 0.3:
        warns.append(f"⚠ 回旋 r_g/Δx={c['r_g_over_dx']:.4f} < 0.3")
    else:
        warns.append(f"✅ 回旋 r_g/Δx={c['r_g_over_dx']:.4f}")

    if c["coll_step"] > 1:
        warns.append(f"❌ 碰撞步 coll_step={c['coll_step']:.2e} > 1")
    elif c["coll_step"] > 0.1:
        warns.append(f"⚠ 碰撞步 coll_step={c['coll_step']:.3f} > 0.1")
    else:
        warns.append(f"✅ 碰撞步 coll_step={c['coll_step']:.4f}")

    if c["τ_coll_vs_τ_sim"] < 0.1:
        warns.append(f"⚡ 强碰撞 τ_coll/τ_sim={c['τ_coll_vs_τ_sim']:.3e}")
    elif c["τ_coll_vs_τ_sim"] < 1:
        warns.append(f"⚡ 有碰撞 τ_coll/τ_sim={c['τ_coll_vs_τ_sim']:.3f}")
    elif c["τ_coll_vs_τ_sim"] < 10:
        warns.append(f"⚡ 弱碰撞 τ_coll/τ_sim={c['τ_coll_vs_τ_sim']:.2f}")
    else:
        warns.append(f"⚡ 无碰撞 τ_coll/τ_sim={c['τ_coll_vs_τ_sim']:.2e}")

    critical = [w for w in warns if w.startswith("❌")]
    return len(critical) == 0, warns


# ============================================================
# 7. 主程序
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BBN-PIC 参数映射与约束分析")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.01, 0.03, 0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()

    epochs = read_cosmocons()
    sigmas = args.sigmas

    # 只分析有物理意义的时期 (T9 > 0.01, Ne > 0)
    relevant = [ep for ep in epochs if ep["Ne_cm3"] > 0 and ep["t9"] > 0.01]

    print("=" * 120)
    print("  BBN → PIC 参数映射与约束分析")
    print(f"  参考点: T_plasma={REF_T_eV/1000:.2f} keV, n_plasma={REF_n_m3:.2e} m⁻³")
    print(f"  CosmoCons.dat 参考点 Ne 内插: T9={REF_T9:.4f}, Ne={REF_Ne_cm3:.4e} cm⁻³")
    print(f"  n_plasma 归一化因子 (sim Ne/BBN Ne) = {NORM_FACTOR:.6e}")
    print(f"  网格: {SIM_DEFAULTS['NX']}³, LX={SIM_DEFAULTS['LX']}, DT={SIM_DEFAULTS['DT']}")
    print("=" * 120)

    for ep in relevant[:15]:
        T_ref = ep["T_eV"]
        n = n_plasma_from_epoch(ep)
        t9 = ep["t9"]

        print(f"\n{'─'*120}")
        print(f"  BBN 时期: T9={t9:.4f}, t={ep['time_s']:.3f} s")
        print(f"  → T_ref={T_ref:.1f} eV ({T_ref/1000:.2f} keV)")
        print(f"  → n_plasma={n:.4e} m⁻³  (Ne_BBN={ep['Ne_cm3']:.4e} cm⁻³)")
        print(f"{'─'*120}")

        print(f"  {'sigma':>8} | {'T_new(keV)':>10} | {'B(T)':>11} | {'λ_D/Δx':>8} | {'r_g/Δx':>8} | "
              f"{'ν_c/ω_pe':>9} | {'coll步':>7} | {'τ_c/τ_s':>9} | {'判据':>6}")
        print(f"  {'─'*8}-+-{'─'*10}-+-{'─'*11}-+-{'─'*8}-+-{'─'*8}-+-"
              f"{'─'*9}-+-{'─'*7}-+-{'─'*9}-+-{'─'*6}")

        for sigma in sigmas:
            c = analyze(ep, sigma)
            if c is None:
                print(f"  {sigma:>8.6f} | {'—':>10s} | {'—':>11s} | {'—':>8s} | {'—':>8s} | "
                      f"{'—':>9s} | {'—':>7s} | {'—':>9s} | {'❌':>6s}")
                continue

            feasible, warns = evaluate(c)
            # 仅显示最关键的判据符号
            if not feasible:
                flag = "❌"
            elif any(w.startswith("⚠") for w in warns):
                flag = "⚠"
            else:
                flag = "✅"

            print(f"  {sigma:>8.6f} | {c['T_new_eV']/1000:>10.4f} | {c['B_T']:>11.4e} | "
                  f"{c['λ_D_over_dx']:>8.4f} | {c['r_g_over_dx']:>8.4f} | "
                  f"{c['ν_coll_over_ω_pe']:>9.4e} | {c['coll_step']:>7.4f} | "
                  f"{c['τ_coll_vs_τ_sim']:>9.4e} | {flag:>6s}")

        # 给出最推荐参数
        print(f"\n  → 推荐参数 (▸ 最宽松的可行 sigma):")
        best_idx = None
        for i, sigma in enumerate(sigmas):
            c = analyze(ep, sigma)
            if c is None:
                break
            feasible, _ = evaluate(c)
            if feasible:
                best_idx = i
            else:
                break
        if best_idx is not None:
            c_best = analyze(ep, sigmas[best_idx])
            print(f"     sigma={sigmas[best_idx]:.4f}: T_new={c_best['T_new_eV']/1000:.2f} keV, "
                  f"B={c_best['B_T']:.3e} T, n={c_best['n_m3']:.3e} m⁻³")
            if best_idx > 0:
                c_cons = analyze(ep, sigmas[0])
                print(f"     保守 sigma={sigmas[0]:.4f}: T_new={c_cons['T_new_eV']/1000:.2f} keV, "
                      f"B={c_cons['B_T']:.3e} T")
        print()

    # 综合建议
    print("\n" + "=" * 80)
    print("  综合建议：BBN 时期 × sigma 参数矩阵")
    print("=" * 80)
    print(f"  {'T9':>6} | {'t(s)':>8} | {'T_ref(keV)':>10} | {'n(m⁻³)':>11} | ", end="")
    for s in sigmas:
        print(f" σ={s:<.4f} |", end="")
    print()
    print(f"  {'─'*6}-+-{'─'*8}-+-{'─'*10}-+-{'─'*11}-+-" + "─" * (len(sigmas) * 10) + "─" * (len(sigmas)))

    for ep in relevant[:10]:
        line = f"  {ep['t9']:>6.4f} | {ep['time_s']:>8.3f} | {ep['T_eV']/1000:>10.2f} | {n_plasma_from_epoch(ep):>11.4e} |"
        for sigma in sigmas:
            c = analyze(ep, sigma)
            if c is None:
                line += f" {'❌':>8s} |"
            else:
                feasible, _ = evaluate(c)
                line += f" {'✅' if feasible else '⚠️' if any(w.startswith('⚠') for w in evaluate(c)[1]) else '❌'}:{c['T_new_eV']/1000:>5.1f} |"
        print(line)


if __name__ == "__main__":
    main()
