#!/usr/bin/env python3
"""
温度 - sigma 参数空间探索工具。

功能：
1. 给定一个基准温度 T_ref 和 target_sigma，计算恒能量分配下的实际温度 T_new 和磁场 B
2. 扫描多个温度、多个 sigma，给出完整的参数空间表格
3. 估算碰撞频率变化（碰撞频率 ∝ n / T^{3/2}）
4. 提示 sigma 上限
"""

import math
import json
import hashlib
from typing import List, Tuple

from scipy.constants import m_e, c, e, mu_0

# ============================================================
# 物理常数
# ============================================================
MC2_J = m_e * c ** 2          # 电子静止能量 (J)
MC2_eV = MC2_J / e            # 电子静止能量 (eV) ≈ 511 keV
MU_0 = mu_0

# ============================================================
# 核心计算
# ============================================================

def calculate_partitioned_energy(T_ref_eV: float, sigma: float, n_plasma: float = 7.28e33):
    """
    恒能量分配模式下，给定 T_ref 和 sigma，计算实际温度和磁场。
    公式推导见 csv_tool_constant_energy.py。
    """
    if sigma < 0:
        raise ValueError(f"sigma 不能为负: {sigma}")

    kb_T_ref_J = T_ref_eV * e

    # T_new (J)
    term1 = (MC2_J + 3.0 * kb_T_ref_J) / (1.0 + sigma)
    kb_T_new_J = (term1 - MC2_J) / 3.0

    if kb_T_new_J < 0:
        sigma_max = (3.0 * kb_T_ref_J) / MC2_J
        return None, None, sigma_max

    T_new_eV = kb_T_new_J / e

    # B_new (T)
    U_p_new = 2 * n_plasma * (MC2_J + 3.0 * kb_T_new_J)
    U_B_new = sigma * U_p_new
    B_new = math.sqrt(2 * MU_0 * U_B_new)

    return T_new_eV, B_new, None


def sigma_max_for_temperature(T_ref_eV: float) -> float:
    """计算给定 T_ref 下恒能量分配的 sigma 理论上限。"""
    return 3.0 * T_ref_eV / MC2_eV


def collision_freq_ratio(T_new_eV: float, T_ref_eV: float) -> float:
    """
    碰撞频率 ∝ n / T^{3/2} (经典)，或更精确地考虑相对论修正。
    这里给出相对于 T_ref 的碰撞频率变化倍数。
    >1 表示碰撞更强，<1 表示碰撞更弱。
    """
    if T_new_eV <= 0 or T_ref_eV <= 0:
        return float('inf')
    return (T_ref_eV / T_new_eV) ** 1.5


def debye_length_ratio(T_new_eV: float, T_ref_eV: float, n: float = 7.28e33) -> float:
    """
    Debye 长度 λ_D = sqrt(ε₀ T / (n e²))。
    相对于 T_ref 的变化倍数。
    T 降低 -> Debye 长度缩短 -> 同一网格能解析更小的尺度。
    """
    if T_new_eV <= 0 or T_ref_eV <= 0:
        return float('inf')
    return math.sqrt(T_new_eV / T_ref_eV)


def thermal_velocity_ratio(T_new_eV: float, T_ref_eV: float) -> float:
    """
    热速度 v_th ∝ sqrt(T/m)，相对论修正前。
    低温时粒子运动变慢→可能需要调整 DT (CFL 条件放宽)。
    """
    if T_new_eV <= 0 or T_ref_eV <= 0:
        return float('inf')
    return math.sqrt(T_new_eV / T_ref_eV)


def gyroradius_ratio(T_new_eV: float, B_new: float, T_ref_eV: float, B_ref: float) -> float:
    """
    回旋半径 r_g = γ m v / (eB) ≈ sqrt(T) / B (非相对论近似)。
    这里给出相对于参考值的变化倍数。
    """
    if B_new <= 0 or B_ref <= 0:
        return float('inf')
    r_g_ratio = math.sqrt(T_new_eV / T_ref_eV) * (B_ref / B_new)
    return r_g_ratio


# ============================================================
# 表格输出
# ============================================================

def print_sigma_scan(T_ref_eV: float, sigmas: List[float], n_plasma: float = 7.28e33):
    """
    对单个 T_ref，扫描一组 sigma 值，打印结果表格。
    """
    smax = sigma_max_for_temperature(T_ref_eV)
    print(f"\n{'='*100}")
    print(f"  基准温度 T_ref = {T_ref_eV:.1f} eV  ({T_ref_eV/1000:.2f} keV)")
    print(f"  数密度 n       = {n_plasma:.3e} m⁻³")
    print(f"  sigma 理论上限  = {smax:.4f}  (T_new → 0)")
    print(f"{'='*100}")
    print(f"{'sigma':>8} | {'T_new(eV)':>12} | {'T_new(keV)':>10} | {'B(T)':>14} | {'ΔT%':>8} | {'ν_coll/ν_ref':>12} | {'v_th/v_ref':>10} | {'r_g/r_ref':>10}")
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*14}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

    # 先算 T_ref 自身的 B_ref (sigma → 0 极限)
    _, B_ref, _ = calculate_partitioned_energy(T_ref_eV, 1e-6, n_plasma)

    for sigma in sigmas:
        result = calculate_partitioned_energy(T_ref_eV, sigma, n_plasma)
        if result[0] is None:
            _, _, actual_smax = result
            print(f"{sigma:>8.4f} | {'—':>12s} | {'—':>10s} | {'—':>14s} | {'—':>8s} | {'—':>12s} | {'—':>10s} | {'—':>10s}  ← 超出上限 (smax={actual_smax:.4f})")
            continue

        T_new_eV, B_new, _ = result
        delta_T_pct = (T_new_eV - T_ref_eV) / T_ref_eV * 100
        coll_ratio = collision_freq_ratio(T_new_eV, T_ref_eV)
        vt_ratio = thermal_velocity_ratio(T_new_eV, T_ref_eV)
        rg_ratio = gyroradius_ratio(T_new_eV, B_new, T_ref_eV, B_ref)

        print(f"{sigma:>8.6f} | {T_new_eV:>12.1f} | {T_new_eV/1000:>10.4f} | {B_new:>14.4e} | {delta_T_pct:>8.4f} | {coll_ratio:>12.4e} | {vt_ratio:>10.6f} | {rg_ratio:>10.6f}")

    # 在 smax 附近选个接近极限的点
    near_limit = smax * 0.95
    if near_limit > 0:
        result = calculate_partitioned_energy(T_ref_eV, near_limit, n_plasma)
        if result[0] is not None:
            T_lim, B_lim, _ = result
            print(f"{'~smax×0.95':>8} | {T_lim:>12.1f} | {T_lim/1000:>10.4f} | {B_lim:>14.4e} | {'':>8s} | {'':>12s} | {'':>10s} | {'':>10s}")
    print()


def compare_temperatures(
    temperatures_eV: List[float],
    sigma: float,
    n_plasma: float = 7.28e33,
):
    """
    固定 sigma，对比不同温度下的参数。
    """
    print(f"\n{'='*100}")
    print(f"  固定 sigma = {sigma}，对比不同温度")
    print(f"{'='*100}")
    print(f"{'T_ref(eV)':>12} | {'T_ref(keV)':>10} | {'T_new(eV)':>12} | {'B(T)':>14} | {'ν_coll/ν_ref(84keV)':>20}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*20}")

    # 以 84.48 keV 为碰撞频率参考
    _, B_84, _ = calculate_partitioned_energy(84480.0, sigma, n_plasma)
    T_ref_84 = 84480.0

    for T_ref in temperatures_eV:
        result = calculate_partitioned_energy(T_ref, sigma, n_plasma)
        if result[0] is None:
            print(f"{T_ref:>12.1f} | {T_ref/1000:>10.4f} | {'—':>12s} | {'—':>14s} | {'—':>20s}")
            continue
        T_new, B, _ = result
        coll_vs_84 = collision_freq_ratio(T_new, T_ref_84)
        print(f"{T_ref:>12.1f} | {T_ref/1000:>10.4f} | {T_new:>12.1f} | {B:>14.4e} | {coll_vs_84:>20.4e}")
    print()


def generate_queue_entries(
    T_ref_eV: float,
    sigma: float,
    base_params: dict,
    n_plasma: float = 7.28e33,
) -> dict:
    """
    生成单个 queue.jsonl 条目参数字典（恒能量分配模式）。
    base_params 是除了 T_plasma_eV/B0/target_sigma 外的固定参数。
    """
    T_new, B, _ = calculate_partitioned_energy(T_ref_eV, sigma, n_plasma)
    if T_new is None:
        return None

    params = dict(base_params)
    params['T_plasma_eV'] = T_new
    params['B0'] = B
    params['target_sigma'] = sigma
    params['_generated_mode'] = "constant_energy_partition"
    params['_T_ref_eV'] = T_ref_eV  # 记录参考温度

    return params


# ============================================================
# 建议的 sigma 扫描列表
# ============================================================

DEFAULT_SIGMAS = [
    0.0001, 0.0005, 0.001, 0.005,
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.08, 0.10,
    0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
]

DEFAULT_TEMPERATURES = [
    84480.0,   # 原始 (84.48 keV)
    58000.0,   # 58 keV
    30000.0,   # 30 keV
    12000.0,   # 12 keV
    5000.0,    # 5 keV
]

# ============================================================
# 主程序
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="温度 - sigma 参数空间探索工具")

    parser.add_argument("--temperatures", type=float, nargs="+", default=None,
                        help="要探索的温度列表 (eV)，默认: 84.5k 58k 30k 12k 5k")
    parser.add_argument("--sigmas", type=float, nargs="+", default=None,
                        help="要探索的 sigma 列表，默认覆盖 0.0001 ~ 0.45")
    parser.add_argument("--fixed-sigma", type=float, default=None,
                        help="固定 sigma 横向对比不同温度")
    parser.add_argument("--n-plasma", type=float, default=7.28e33,
                        help="等离子体数密度 (m⁻³)")
    parser.add_argument("--generate-json", type=str, default=None,
                        help="输出路径：为指定温度生成 queue.jsonl 内容")
    parser.add_argument("--grid", type=str, default="LX100LY100LZ100_NX256NY256NZ256",
                        help="网格标识，用于 --generate-json 的 output_dir 命名")

    args = parser.parse_args()

    temperatures = args.temperatures or DEFAULT_TEMPERATURES
    sigmas = args.sigmas or DEFAULT_SIGMAS

    print("=" * 100)
    print("  等离子体恒能量分配 — 参数空间探索")
    print(f"  n_plasma = {args.n_plasma:.3e} m⁻³")
    print(f"  m_e·c²  = {MC2_eV:.1f} eV")
    print("=" * 100)

    # 1. 对每个温度完整扫描 sigma
    for T_ref in temperatures:
        print_sigma_scan(T_ref, sigmas, args.n_plasma)

    # 2. 如果指定了固定 sigma，横向对比
    if args.fixed_sigma is not None:
        compare_temperatures(temperatures, args.fixed_sigma, args.n_plasma)

    # 3. 尝试生成具体建议
    print("\n" + "=" * 70)
    print("  综合建议：推荐的运行参数组合")
    print("=" * 70)
    print(f"{'T_ref':>8} | {'sigma':>8} | {'T_new(keV)':>12} | {'B(T)':>14} | {'运行时间(256³/100ppc)':>24}")
    print(f"{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*14}-+-{'-'*24}")

    # 对每个温度，推荐 3-4 个有代表性的 sigma 点
    representative = {
        84480.0: [0.01, 0.05, 0.10, 0.20, 0.30, 0.40],
        58000.0: [0.01, 0.05, 0.10, 0.20, 0.30],
        30000.0: [0.01, 0.05, 0.10, 0.15],
        12000.0: [0.01, 0.03, 0.05, 0.06],
        5000.0:  [0.005, 0.01, 0.02, 0.025],
    }
    for T_ref in temperatures:
        reps = representative.get(T_ref, [0.01, 0.05])
        for sigma in reps:
            result = calculate_partitioned_energy(T_ref, sigma, args.n_plasma)
            if result[0] is None:
                continue
            T_new, B, _ = result
            if T_ref == max(temperatures) and sigma <= 0.052:
                runtime = "≈ 72 min (已有数据)"
            else:
                runtime = "≈ 72 min (需运行)"
            print(f"{T_ref/1000:>8.2f}k | {sigma:>8.6f} | {T_new/1000:>12.4f} | {B:>14.4e} | {runtime:>24}")

    # 4. 如果指定了 --generate-json，输出 JSONL
    if args.generate_json:
        base = {
            "DT": 0.2, "LT": 1000.0,
            "LX": 100.0, "LY": 100.0, "LZ": 100.0,
            "NPPC": 100, "NX": 256, "NY": 256, "NZ": 256,
            "dim": 2,
            "field_total_step": 10, "particle_total_step": 10,
            "enable_collision": True, "enable_qed": False,
            "n_photon_to_plasma_ratio": 3.93,
            "n_plasma": args.n_plasma,
            "beam_energy_eV": 84480.0, "beam_fraction": 0.0,
            "B_field_type": "multi_gaussian",
            "Bg_ratio": -1.0, "dB_ratio": 0.0,
            "num_gaussians": 100,
            "gaussian_width_de_ratio": 2.5,
        }

        target_T = float(args.generate_json)
        entries = []
        reps = representative.get(target_T, [0.01, 0.05])
        for sigma in reps:
            params = generate_queue_entries(target_T, sigma, base, args.n_plasma)
            if params is None:
                continue
            hash_input = json.dumps(params, sort_keys=True).encode()
            short_hash = hashlib.sha256(hash_input).hexdigest()[:12]
            sigma_str = f"{sigma:.4f}"
            dir_name = f"{args.grid}_Tref{target_T/1000:.0f}k_sigma{sigma_str}_{short_hash}"
            entry = {
                "params": params,
                "output_dir": f"sim_results/{dir_name}"
            }
            entries.append(entry)

        out_path = f"queue_Tref{target_T/1000:.0f}k.jsonl"
        with open(out_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"\n[✔] 已生成队列文件: {out_path} ({len(entries)} 个任务)")


if __name__ == "__main__":
    main()
